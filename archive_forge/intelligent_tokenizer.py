"""
eidos_framework/intelligent_tokenizer.py

This module implements a truly "intelligent" adaptive tokenizer—a single unified system 
that merges basic character tokenization, dynamic learning, and an advanced vocabulary builder 
with subword merging, transformer integration, and persistence. It is extensible, modular, robust, 
and resistant to interruptions.
"""

from __future__ import annotations
import numpy as np
import os
import pickle
import hashlib
import warnings
import threading
import time
import unicodedata
import importlib.metadata
import logging
import nltk  # type: ignore[import]
from nltk.corpus import words as nltk_words  # type: ignore[import]
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import sentencepiece as spm  # type: ignore[import]
import concurrent.futures
from collections import Counter, defaultdict
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments  # type: ignore[import]
from colorama import Fore, Back, Style, init
from typing import Dict, List, Set, Tuple, Any, Literal, Optional, Iterable, DefaultDict
from wiktionaryparser import WiktionaryParser  # type: ignore[import]

# -----------------------------------------------------------------------------
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration Module
@dataclass
class TokenizerConfig:
    normalization_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    special_tokens: Optional[Tuple[str, ...]] = None
    unicode_strategy: Optional[str] = None
    unicode_blocks: Optional[Iterable[Tuple[int, int]]] = None
    category_profile: Optional[str] = None
    technical_categories: Optional[Set[str]] = None
    control_chars: Optional[Iterable[int]] = None
    custom_chars: Optional[Iterable[str]] = None
    sort_mode: Literal["unicode", "frequency", "custom"] = "unicode"
    dynamic_rebuild: bool = True
    persistence_prefix: Optional[str] = None
    modes: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Plugin Interfaces Module
class BasePlugin:
    """Abstract base class for tokenizer plugins."""

    def get_chars(self) -> Set[str]:
        raise NotImplementedError("Plugin must implement get_chars()")

    def get_processor(self) -> Any:
        raise NotImplementedError("Plugin must implement get_processor()")

    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError("Plugin must implement serialize()")

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> BasePlugin:
        raise NotImplementedError("Plugin must implement deserialize()")


# -----------------------------------------------------------------------------
# Unicode Utilities Module
class UnicodeUtils:
    @staticmethod
    @lru_cache(maxsize=None)
    def compute_chars_for_categories(categories: frozenset) -> Set[str]:
        result = set()
        for code in range(0x10FFFF + 1):
            try:
                char = chr(code)
            except (ValueError, OverflowError):
                continue
            cat = unicodedata.category(char)
            if any(
                (len(c) == 1 and cat.startswith(c)) or (len(c) == 2 and cat == c)
                for c in categories
            ):
                result.add(char)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def compute_intervals_for_category(category: str) -> List[Tuple[int, int]]:
        intervals: List[Tuple[int, int]] = []
        start = None
        prev = None
        for code in range(0x10FFFF + 1):
            try:
                ch = chr(code)
            except (ValueError, OverflowError):
                continue
            cat = unicodedata.category(ch)
            match = (len(category) == 1 and cat.startswith(category)) or (
                len(category) == 2 and cat == category
            )
            if match:
                if start is None:
                    start = code
                    prev = code
                elif prev is not None and code == prev + 1:
                    prev = code
                else:
                    if start is not None and prev is not None:
                        intervals.append((int(start), int(prev)))
                    start = code
                    prev = code
            else:
                if start is not None and prev is not None:
                    intervals.append((int(start), int(prev)))
                    start = None
                    prev = None
        if start is not None and prev is not None:
            intervals.append((int(start), int(prev)))
        if not intervals:
            intervals.append((0, 0x10FFFF))
        return intervals


# -----------------------------------------------------------------------------
# Plugin Manager
class PluginManager:
    """
    Discovers and attaches tokenizer plugins.
    """

    def __init__(self, modes: Dict[str, Any]):
        self.modes = modes
        self.plugins: Dict[str, BasePlugin] = {}
        self.initialize_plugins()

    def initialize_plugins(self) -> None:
        try:
            eps = importlib.metadata.entry_points()
            if isinstance(eps, dict):
                plugin_entries = eps.get("my_tokenizer.plugins", [])
            else:
                plugin_entries = list(eps.select(group="my_tokenizer.plugins"))
            for ep in plugin_entries:
                plugin_instance = ep.load()()
                if isinstance(plugin_instance, BasePlugin):
                    self.plugins[ep.name] = plugin_instance
                else:
                    logger.warning(
                        f"Plugin {ep.name} does not implement BasePlugin interface."
                    )
        except Exception as e:
            logger.warning(f"Dynamic plugin discovery failed: {e}")
        if "plugins" in self.modes:
            for name, plugin in self.modes["plugins"].items():
                if isinstance(plugin, BasePlugin):
                    self.plugins[name] = plugin

    def attach_plugins(self, target: Any) -> None:
        for name, plugin in self.plugins.items():
            try:
                processor = plugin.get_processor()
                setattr(target, f"apply_{name}", processor)
            except Exception as e:
                logger.error(f"Error attaching plugin '{name}': {e}")


# -----------------------------------------------------------------------------
# Persistence Manager
class PersistenceManager:
    """
    Saves/loads vocabulary state.
    """

    def __init__(self, persistence_prefix: Optional[str]):
        self.persistence_prefix = persistence_prefix

    def load_vocabulary(self, config_hash: str) -> Optional[Dict[str, Any]]:
        vocab_file = f"{self.persistence_prefix}.vocab"
        config_hash_file = f"{self.persistence_prefix}.config_hash"
        try:
            if os.path.exists(vocab_file) and os.path.exists(config_hash_file):
                with open(config_hash_file, "r") as f:
                    stored_hash = f.read().strip()
                if stored_hash == config_hash:
                    with open(vocab_file, "rb") as f_vocab:
                        return pickle.load(f_vocab)
        except (OSError, pickle.PickleError) as e:
            logger.warning(f"Error loading vocabulary: {e}")
        return None

    def save_vocabulary(
        self, config_hash: str, vocab: Dict[str, int], inverse_vocab: Dict[int, str]
    ) -> None:
        vocab_file = f"{self.persistence_prefix}.vocab"
        config_hash_file = f"{self.persistence_prefix}.config_hash"
        for attempt in range(3):
            try:
                with open(vocab_file, "wb") as f_vocab:
                    pickle.dump(
                        {"vocab": vocab, "inverse_vocab": inverse_vocab}, f_vocab
                    )
                with open(config_hash_file, "w") as f_config:
                    f_config.write(config_hash)
                logger.info(f"{Fore.CYAN}Saved vocabulary to persistence.")
                return
            except (OSError, pickle.PickleError) as e:
                logger.error(f"Attempt {attempt+1} error saving vocab: {e}")
        warnings.warn("Failed to save vocabulary after 3 attempts.")


# -----------------------------------------------------------------------------
# LRU Cache and Trie for efficient token matching
class LRUCache(dict):
    def __init__(self, maxsize: int = 4096, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        elif len(self) >= self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
        super().__setitem__(key, value)

    def clear(self):
        super().clear()


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.token: Optional[str] = None


class Trie:
    """
    Trie structure for fast greedy prefix matching.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self.search_cache = LRUCache(maxsize=4096)

    def insert(self, token: str) -> None:
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.token = token
        self.search_cache.clear()
        logger.debug(f"Inserted token into trie: {token}")

    def search_longest(self, text: str, start_index: int) -> str:
        cache_key = (text, start_index)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        node = self.root
        longest = None
        i = start_index
        while i < len(text) and text[i] in node.children:
            node = node.children[text[i]]
            if node.is_end:
                longest = node.token
            i += 1
        result = longest if longest is not None else ""
        self.search_cache[cache_key] = result
        return result


# -----------------------------------------------------------------------------
# Text Processing and N-Gram extraction modules
class TextProcessor:
    def __init__(
        self,
        mode: str,
        lowercase: bool,
        nlp: Optional[Language],
        tokenize_cache: LRUCache,
    ):
        self.mode = mode
        self.lowercase = lowercase
        self.nlp = nlp
        self.tokenize_cache = tokenize_cache

    def normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def get_doc(self, text: str) -> Optional[Doc]:
        if text in self.tokenize_cache:
            return self.tokenize_cache[text]
        try:
            doc = self.nlp(text) if self.nlp else None
        except Exception as e:
            logger.warning(f"Tokenization error for text {text!r}: {e}")
            doc = None
        if doc:
            self.tokenize_cache[text] = doc
        return doc


class NGramExtractor:
    def __init__(self, max_ngram: int, cache: LRUCache):
        self.max_ngram = max_ngram
        self.cache = cache

    def extract(self, tokens: List[str], joiner: str) -> Counter[str]:
        """
        Helper method to extract n-grams from a list of tokens.
        """
        cache_key = (joiner, tuple(tokens))
        if cache_key in self.cache:
            return self.cache[cache_key]
        counts: Counter[str] = Counter()
        token_count: int = len(tokens)
        for ngram_size in range(2, self.max_ngram + 1):
            if token_count >= ngram_size:
                for i in range(token_count - ngram_size + 1):
                    ngram: str = joiner.join(tokens[i : i + ngram_size])
                    counts[ngram] += 1
        self.cache[cache_key] = counts
        return counts

    def process_text(self, text: str, text_processor: TextProcessor) -> Counter[str]:
        """
        Process a text sample to extract n-gram counts.
        """
        local_counts: Counter[str] = Counter()
        proc_text: str = text_processor.normalize(text)
        if text_processor.mode == "word" and text_processor.nlp:
            doc: Optional[Doc] = text_processor.get_doc(proc_text)
            if doc is not None:
                word_tokens = [token.text for token in doc if not token.is_space]
            else:
                logger.warning(
                    "spaCy tokenization error in parallel processing; falling back to simple split."
                )
                word_tokens = proc_text.split()
            char_tokens: List[str] = list(proc_text)
            local_counts.update(self.extract(word_tokens, " "))
            local_counts.update(self.extract(char_tokens, ""))
        else:
            tokens: List[str] = list(proc_text)
            local_counts.update(self.extract(tokens, ""))
        return local_counts


class MetadataPopulator:
    def __init__(
        self,
        nlp: Optional[Language],
        transformer_model: Optional[Any],
        transformer_tokenizer: Optional[Any],
        embedding_cache: LRUCache,
    ):
        self.nlp = nlp
        self.transformer_model = transformer_model
        self.transformer_tokenizer = transformer_tokenizer
        self.embedding_cache = embedding_cache

    def populate(
        self,
        token: str,
        existing_metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Populate optional metadata for a token from various linguistic resources.
        """
        metadata: Dict[str, Any] = (
            existing_metadata if existing_metadata is not None else {}
        )
        # WordNet definitions:
        if token.isalpha():
            synsets = nltk.corpus.wordnet.synsets(token)
            definitions = [syn.definition() for syn in synsets if syn]
            if definitions:
                metadata.setdefault("wordnet_definitions", definitions)
        # Wiktionary integration using wiktionaryparser.
        try:
            from wiktionaryparser import WiktionaryParser  # type: ignore

            parser = WiktionaryParser()
            wikidata = parser.fetch(token)
            if wikidata:
                definitions = []
                for entry in wikidata:
                    for definition in entry.get("definitions", []):
                        if "text" in definition:
                            definitions.extend(definition["text"])
                if definitions:
                    metadata["wiktionary"] = definitions
        except Exception as e:
            logger.error("Error fetching Wiktionary data for token %s: %s", token, e)
        # Morphological analysis using spaCy.
        if self.nlp:
            doc: Optional[Doc] = self.nlp(token)
            if doc:
                for tok in doc:
                    metadata["morphology"] = tok.morph.to_dict()
        # Populate transformer-based embeddings.
        if (
            self.transformer_model is not None
            and self.transformer_tokenizer is not None
        ):
            if token not in self.embedding_cache:
                inputs = self.transformer_tokenizer(token, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                self.embedding_cache[token] = embedding
            metadata["embedding"] = self.embedding_cache[token]
        else:
            logger.warning(
                "Transformer objects are not initialized; skipping embedding population for token: %s",
                token,
            )
        logger.debug("Populated metadata for token: %s", token)
        return metadata


# -----------------------------------------------------------------------------
# Transformer Manager for contextual embedding operations
DEFAULT_TRAINING_ARGS = TrainingArguments(
    output_dir="~/Development/qwen_finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=10,
    logging_dir="~/Development/logs",
    disable_tqdm=True,
)


class TransformerManager:
    def __init__(
        self,
        transformer_tokenizer: Any,
        transformer_model: Any,
        device: torch.device,
        embedding_cache: LRUCache,
    ):
        self.tokenizer = transformer_tokenizer
        self.model = transformer_model
        self.device = device
        self.embedding_cache = embedding_cache

    def load_model(self, model_name: str) -> None:
        """
        Load and initialize the transformer model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.train()  # Switch to training mode
        self.model.to(self.device)
        logger.info("Transformer model loaded: %s", model_name)

    def async_train(self, inputs: Any, dataset: Any) -> None:
        """
        Asynchronously fine-tune the transformer model on the provided dataset.
        """
        trainer = Trainer(
            model=self.model,
            args=DEFAULT_TRAINING_ARGS,
            train_dataset=dataset,
        )
        trainer.train()
        logger.info("Transformer model fine-tuned.")


# -----------------------------------------------------------------------------
# Corpus Manager for dynamic corpus updates.
class CorpusManager:
    def __init__(self, corpus_file: Path):
        self.dynamic_corpus: List[str] = []
        self.dynamic_corpus_file: Path = corpus_file
        self.lock = threading.Lock()

    def update(self, new_text: str) -> None:
        """
        Update the dynamic corpus file with new text.
        """
        with self.lock:
            self.dynamic_corpus.append(new_text)
        try:
            with self.dynamic_corpus_file.open("a", encoding="utf-8") as f:
                f.write(new_text + "\n")
            logger.info("Dynamic corpus updated with new text.")
        except Exception as e:
            logger.error("Error updating dynamic corpus file: %s", e)
            raise


# -----------------------------------------------------------------------------
# AutoSaver for periodic persistence of the vocabulary.
class AutoSaver:
    def __init__(
        self,
        builder: AdvancedVocabularyBuilder,
        auto_save_interval: int,
        auto_save_file: Path,
    ):
        self.builder = builder
        self.auto_save_interval = auto_save_interval
        self.auto_save_file = auto_save_file
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.thread.start()

    def _auto_save_loop(self) -> None:
        """
        Periodically save the vocabulary using the builder's save function.
        """
        while not self.shutdown_event.is_set():
            time.sleep(self.auto_save_interval)
            try:
                self.builder.save_vocab(str(self.auto_save_file))
                logger.info("Auto-saved vocabulary to %s", self.auto_save_file)
            except Exception as e:
                logger.error("Auto-save failed: %s", e)

    def stop(self) -> None:
        """
        Stop the auto-saving thread.
        """
        self.shutdown_event.set()
        self.thread.join(timeout=5)


# -----------------------------------------------------------------------------
# Modular Component: Subword Merging (BPE, Contextual BPE, SentencePiece, Unigram LM)
class SubwordMerger:
    def __init__(self, vocab_builder: AdvancedVocabularyBuilder):
        """
        Initialize with an instance of AdvancedVocabularyBuilder.
        """
        self.builder = vocab_builder

    def apply_bpe(self, num_merges: int = 500) -> None:
        """
        Apply a Byte-Pair Encoding (BPE) inspired subword merging algorithm with enhanced logging and robustness.
        """
        logger.info("Starting BPE subword merging process...")
        bpe_vocab: Dict[str, int] = {
            token: freq
            for token, freq in self.builder.multi_token_vocab.items()
            if token.isalpha() and len(token) > 1
        }
        bpe_tokens: Dict[Tuple[str, ...], int] = {
            tuple(token): freq for token, freq in bpe_vocab.items()
        }
        iteration: int = 0

        for _ in tqdm(range(num_merges), desc="BPE iterations", leave=False):
            iteration += 1
            pair_freqs: DefaultDict[Tuple[str, str], int] = defaultdict(int)
            for token_tuple, freq in bpe_tokens.items():
                for j in range(len(token_tuple) - 1):
                    pair = (token_tuple[j], token_tuple[j + 1])
                    pair_freqs[pair] += freq
            if not pair_freqs:
                logger.debug("No more pairs to merge in BPE.")
                break
            max_pair_freq = max(pair_freqs.values())
            merge_threshold = max(2, int(max_pair_freq * 0.1))
            most_frequent_pair = max(pair_freqs, key=lambda x: pair_freqs[x])
            if pair_freqs[most_frequent_pair] < merge_threshold:
                logger.debug(
                    "BPE stopping: Frequency %s of pair %s is below the merge threshold %s.",
                    pair_freqs[most_frequent_pair],
                    most_frequent_pair,
                    merge_threshold,
                )
                break
            logger.debug(
                "BPE iteration %d: Merging pair %s with frequency %s.",
                iteration,
                most_frequent_pair,
                pair_freqs[most_frequent_pair],
            )
            new_bpe_tokens: Dict[Tuple[str, ...], int] = {}
            for token_tuple, freq in bpe_tokens.items():
                new_token: List[str] = []
                j = 0
                while j < len(token_tuple):
                    if (
                        j < len(token_tuple) - 1
                        and (token_tuple[j], token_tuple[j + 1]) == most_frequent_pair
                    ):
                        new_token.append(token_tuple[j] + token_tuple[j + 1])
                        j += 2
                    else:
                        new_token.append(token_tuple[j])
                        j += 1
                new_bpe_tokens[tuple(new_token)] = (
                    new_bpe_tokens.get(tuple(new_token), 0) + freq
                )
            bpe_tokens = new_bpe_tokens

        for token_tuple, freq in bpe_tokens.items():
            merged_token: str = "".join(token_tuple)
            if merged_token not in self.builder.multi_token_vocab:
                self.builder._add_token(merged_token, freq)
                logger.info("BPE merged token added: %s", merged_token)
        logger.info("BPE subword merging process completed.")

    def apply_contextual_bpe(self, num_merges: int = 300) -> None:
        """
        Apply an enhanced BPE merging that considers contextual similarity.
        """
        logger.info("Starting contextual BPE merging process...")
        bpe_vocab: Dict[str, int] = {
            token: freq
            for token, freq in self.builder.multi_token_vocab.items()
            if token.isalpha() and len(token) > 1
        }
        bpe_tokens: Dict[Tuple[str, ...], int] = {
            tuple(token): freq for token, freq in bpe_vocab.items()
        }
        iteration: int = 0

        for _ in tqdm(range(num_merges), desc="Contextual BPE iterations", leave=False):
            iteration += 1
            pair_metrics: DefaultDict[Tuple[str, str], float] = defaultdict(float)
            for token_tuple, freq in bpe_tokens.items():
                for j in range(len(token_tuple) - 1):
                    pair = (token_tuple[j], token_tuple[j + 1])
                    metric: float = float(freq)
                    if self.builder.transformer_manager.model is not None:

                        emb1 = self.builder.transformer_manager.embedding_cache.get(
                            token_tuple[j]
                        )
                        emb2 = self.builder.transformer_manager.embedding_cache.get(
                            token_tuple[j + 1]
                        )
                        if emb1 is None:
                            if self.builder.transformer_manager.tokenizer is None:
                                logger.warning(
                                    "Transformer tokenizer is not initialized; skipping contextual BPE for pair: %s",
                                    pair,
                                )
                            else:
                                inputs = self.builder.transformer_manager.tokenizer(
                                    token_tuple[j], return_tensors="pt"
                                )
                                with torch.no_grad():
                                    outputs = self.builder.transformer_manager.model(
                                        **inputs
                                    )
                                emb1 = (
                                    outputs.last_hidden_state.mean(dim=1)
                                    .squeeze()
                                    .tolist()
                                )
                                self.builder.transformer_manager.embedding_cache[token_tuple[j]] = emb1  # type: ignore
                        if emb2 is None:
                            if self.builder.transformer_manager.tokenizer is None:
                                logger.warning(
                                    "Transformer tokenizer is not initialized; skipping contextual BPE for pair: %s",
                                    pair,
                                )
                            else:
                                inputs = self.builder.transformer_manager.tokenizer(
                                    token_tuple[j + 1], return_tensors="pt"
                                )
                                with torch.no_grad():
                                    outputs = self.builder.transformer_manager.model(
                                        **inputs
                                    )
                                emb2 = (
                                    outputs.last_hidden_state.mean(dim=1)
                                    .squeeze()
                                    .tolist()
                                )
                                self.builder.transformer_manager.embedding_cache[token_tuple[j + 1]] = emb2  # type: ignore
                        emb1_np = np.array(emb1)
                        emb2_np = np.array(emb2)
                        cosine_sim = np.dot(emb1_np, emb2_np) / (
                            np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np) + 1e-9
                        metric *= cosine_sim
                    pair_metrics[pair] += metric
            if not pair_metrics:
                logger.debug("No more pairs to merge in contextual BPE.")
                break
            max_metric = max(pair_metrics.values())
            merge_threshold = max(2.0, max_metric * 0.1)
            most_promising_pair = max(pair_metrics, key=lambda x: pair_metrics[x])
            if pair_metrics[most_promising_pair] < merge_threshold:
                logger.debug(
                    "Contextual BPE stopping: Metric %.2f for pair %s is below threshold %.2f.",
                    pair_metrics[most_promising_pair],
                    most_promising_pair,
                    merge_threshold,
                )
                break
            logger.debug(
                "Contextual BPE iteration %d: Merging pair %s with metric %.2f.",
                iteration,
                most_promising_pair,
                pair_metrics[most_promising_pair],
            )
            new_bpe_tokens: Dict[Tuple[str, ...], int] = {}
            for token_tuple, freq in bpe_tokens.items():
                new_token: List[str] = []
                j = 0
                while j < len(token_tuple):
                    if (
                        j < len(token_tuple) - 1
                        and (token_tuple[j], token_tuple[j + 1]) == most_promising_pair
                    ):
                        new_token.append(token_tuple[j] + token_tuple[j + 1])
                        j += 2
                    else:
                        new_token.append(token_tuple[j])
                        j += 1
                new_bpe_tokens[tuple(new_token)] = (
                    new_bpe_tokens.get(tuple(new_token), 0) + freq
            bpe_tokens = new_bpe_tokens

        for token_tuple, freq in bpe_tokens.items():
            merged_token: str = "".join(token_tuple)
            if merged_token not in self.builder.multi_token_vocab:
                self.builder._add_token(merged_token, freq)
                logger.info("Contextual BPE merged token added: %s", merged_token)
        logger.info("Contextual BPE merging process completed.")

    def apply_sentencepiece(self) -> None:
        """
        Train and apply a SentencePiece model on the aggregated dynamic corpus.
        """
        corpus_file: str = str(self.builder.corpus_manager.dynamic_corpus_file)
        logger.info(
            "Training SentencePiece model on dynamic corpus from %s...", corpus_file
        )
        model_prefix: str = "spm_model_dynamic"
        try:
            spm.SentencePieceTrainer.train(  # type: ignore
                input=corpus_file,
                model_prefix=model_prefix,
                vocab_size=32000,
                character_coverage=1.0,
                model_type="bpe",
            )
        except Exception as e:
            logger.error("Error training SentencePiece model: %s", e)
            return
        sp = spm.SentencePieceProcessor()  # type: ignore
        sp.load(f"{model_prefix}.model")  # type: ignore
        sp_vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]  # type: ignore
        for token in sp_vocab:
            if (
                token not in self.builder.multi_token_vocab
                and token.isalpha()
                and len(token) > 3
            ):
                self.builder._add_token(token, 1)
                logger.info("SentencePiece merged token added: %s", token)

    def apply_unigram_lm(self, num_merges: int = 500) -> None:
        """
        Train and apply a Unigram LM model on the aggregated dynamic corpus.
        """
        corpus_file: str = str(self.builder.corpus_manager.dynamic_corpus_file)
        logger.info(
            "Training Unigram LM model on dynamic corpus from %s...", corpus_file
        )
        model_prefix: str = "unigram_model_dynamic"
        try:
            spm.SentencePieceTrainer.train(  # type: ignore
                input=corpus_file,
                model_prefix=model_prefix,
                vocab_size=25,
                character_coverage=1.0,
                model_type="unigram",
            )
        except Exception as e:
            logger.error("Error training Unigram LM model: %s", e)
            return
        sp = spm.SentencePieceProcessor()  # type: ignore
        sp.load(f"{model_prefix}.model")  # type: ignore
        sp_vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]  # type: ignore
        for token in sp_vocab:
            if (
                token not in self.builder.multi_token_vocab
                and token.isalpha()
                and len(token) > 4
            ):
                self.builder._add_token(token, 1)
                logger.info("Unigram LM merged token added: %s", token)

    def apply_advanced_merging(self) -> None:
        """
        Apply a layered merging process combining BPE, contextual BPE, SentencePiece, and Unigram LM.
        """
        logger.info("Starting advanced merging processes...")
        if self.builder.use_advanced:
            self.builder._ensure_nltk_resources()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                futures.append(
                    executor.submit(self.builder.enhance_vocabulary_with_wordnet)
                )
                futures.append(
                    executor.submit(self.builder.enhance_vocabulary_with_nltk_words)
                )
                futures.append(
                    executor.submit(self.builder.enhance_vocabulary_with_wiktionary)
                )
                concurrent.futures.wait(futures)
        self.apply_bpe(num_merges=500)
        self.apply_contextual_bpe(num_merges=300)
        self.apply_sentencepiece()
        self.apply_unigram_lm()
        logger.info("Advanced merging processes completed.")


# -----------------------------------------------------------------------------
# Advanced Vocabulary Builder (combining n-gram, advanced merging, transformer & metadata)
class AdvancedVocabularyBuilder:
    def __init__(
        self,
        tokenizer: CharTokenizer,
        min_count: int = 5,
        max_ngram: int = 5,
        mode: str = "character",
        lowercase: bool = True,
        use_advanced: bool = True,
    ):
        # Validate essential parameters and set defaults/fallbacks.
        if tokenizer is None:
            raise ValueError("A valid 'tokenizer' instance must be provided.")
        if min_count < 1:
            raise ValueError("min_count must be at least 1.")
        if max_ngram < 2:
            raise ValueError("max_ngram must be at least 2.")

        self.tokenizer = tokenizer
        self.min_count = min_count
        self.max_ngram = max_ngram
        self.mode = mode
        self.lowercase = lowercase
        self.use_advanced = use_advanced
        self.ngram_counts: Counter[str] = Counter()
        self.multi_token_vocab: Dict[str, int] = {}
        self.token_metadata: Dict[str, Dict[str, Any]] = {}

        self.nlp: Optional[Language] = None
        if self.mode == "word":
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.debug("spaCy model loaded.")
            except Exception as e:
                logger.error("Error loading spaCy model: %s", e)
                raise RuntimeError(
                    "spaCy must be installed and en_core_web_sm available."
                ) from e

        self.trie = Trie()
        self.tokenize_cache = LRUCache(maxsize=4096)
        self.ngram_extraction_cache = LRUCache(maxsize=4096)
        self.text_processor = TextProcessor(
            self.mode, self.lowercase, self.nlp, self.tokenize_cache
        )
        self.ngram_extractor = NGramExtractor(
            self.max_ngram, self.ngram_extraction_cache
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_manager = TransformerManager(
            transformer_tokenizer=AutoTokenizer.from_pretrained(
                "distilbert-base-uncased"
            ),
            transformer_model=AutoModel.from_pretrained("distilbert-base-uncased"),
            device=device,
            embedding_cache=LRUCache(maxsize=4096),
        )
        self.transformer_manager.model.train()
        self.transformer_manager.model.to(device)
        logger.info("Transformer model loaded: distilbert-base-uncased")
        self.metadata_populator = MetadataPopulator(
            self.nlp,
            self.transformer_manager.model,
            self.transformer_manager.tokenizer,
            self.transformer_manager.embedding_cache,
        )
        self.corpus_manager = CorpusManager(Path("~/dynamic_corpus.txt").expanduser())
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.auto_save_interval = 60
        # Use a fallback default for persistence_prefix if not set.
        persistence_prefix = getattr(tokenizer.config, "persistence_prefix", "default")
        self.auto_save_file = Path(f"{persistence_prefix}.vocab")
        self.auto_saver = AutoSaver(self, self.auto_save_interval, self.auto_save_file)

        # Initialize the vocabulary lock to ensure thread-safe operations.
        self._vocab_lock = threading.Lock()

        # Attempt to load previous autosave state if available.
        if self.auto_save_file.exists():
            try:
                self.load_vocab(str(self.auto_save_file))
                logger.info(
                    "Loaded vocabulary from autosave file: %s", self.auto_save_file
                )
            except Exception as exc:
                logger.error(
                    "Failed to load autosave state from %s: %s",
                    self.auto_save_file,
                    exc,
                )

        self.subword_merger = SubwordMerger(self)

    def learn_and_update(self) -> None:
        """
        Perform a comprehensive vocabulary update using all available NLP resources,
        transformer embeddings, and n-gram enhancements.
        """
        logger.info("Starting comprehensive learn and update process...")
        tasks = []
        if self.corpus_manager.dynamic_corpus:
            tasks.append(
                self.executor.submit(
                    self.build_from_corpus, self.corpus_manager.dynamic_corpus
                )
            )
        tasks.append(self.executor.submit(self.enhance_vocabulary_with_wordnet))
        tasks.append(self.executor.submit(self.enhance_vocabulary_with_nltk_words))
        tasks.append(self.executor.submit(self.enhance_vocabulary_with_wiktionary))
        tasks.append(self.executor.submit(self.apply_advanced_merging))
        concurrent.futures.wait(tasks)
        logger.info("Advanced vocabulary update started...")
        # Merge new n-gram tokens into the main vocabulary.
        for token in list(self.multi_token_vocab.keys()):
            if token not in self.tokenizer.vocab:
                self.tokenizer.add_token(token)
                logger.info("Added advanced token: %s", token)
        # (Optionally, additional advanced merging such as contextual BPE can run here.)
        # Auto-save is performed automatically via AutoSaver.
        logger.info("Advanced vocabulary update complete.")
        logger.info("Comprehensive learn and update process completed.")

    def _add_token(self, token: str, count: int) -> None:
        # Protect modifications with the vocabulary lock.
        with self._vocab_lock:
            if token in self.multi_token_vocab:
                return  # Avoid duplicate tokens.
            self.multi_token_vocab[token] = count
        self.trie.insert(token)
        if hasattr(self.tokenizer, "add_token"):
            self.tokenizer.add_token(token)
        if self.use_advanced:
            with self._vocab_lock:
                self.token_metadata[token] = self.metadata_populator.populate(token)

    def _ensure_nltk_resources(self) -> None:
        for resource in ["wordnet", "words", "punkt"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource)

    def enhance_vocabulary_with_wordnet(self) -> None:
        logger.info("Enhancing vocabulary with WordNet...")
        for token in list(self.multi_token_vocab.keys()):
            if token.isalpha():
                synsets = nltk.corpus.wordnet.synsets(token)
                for syn in synsets:
                    for lemma in syn.lemma_names():
                        lemma_lower = lemma.lower()
                        if lemma_lower not in self.multi_token_vocab:
                            self._add_token(lemma_lower, 1)
                            logger.info("Added WordNet lemma: %s", lemma_lower)

    def enhance_vocabulary_with_nltk_words(self) -> None:
        logger.info("Enhancing vocabulary with NLTK words...")
        word_list = set(w.lower() for w in nltk_words.words())
        for word in word_list:
            if word not in self.multi_token_vocab:
                self._add_token(word, 1)
        logger.info("Enhanced vocabulary with NLTK words.")

    def enhance_vocabulary_with_wiktionary(self) -> None:
        logger.info("Enhancing vocabulary with Wiktionary...")
        for token in list(self.multi_token_vocab.keys()):
            if token.isalpha():
                meta = self.token_metadata.get(token, {})
                if "wiktionary" not in meta:
                    enriched_data = self.metadata_populator.populate(
                        token, source="wiktionary"
                    )
                    with self._vocab_lock:
                        self.token_metadata[token] = enriched_data
                    logger.info("Enhanced token with Wiktionary: %s", token)

    def apply_bpe(self, num_merges: int = 500) -> None:
        logger.info("Delegating BPE merging to SubwordMerger...")
        self.subword_merger.apply_bpe(num_merges)

    def apply_contextual_bpe(self, num_merges: int = 300) -> None:
        logger.info("Delegating contextual BPE merging to SubwordMerger...")
        self.subword_merger.apply_contextual_bpe(num_merges)

    def apply_sentencepiece(self) -> None:
        logger.info("Delegating SentencePiece merging to SubwordMerger...")
        self.subword_merger.apply_sentencepiece()

    def apply_unigram_lm(self, num_merges: int = 500) -> None:
        logger.info("Delegating Unigram LM merging to SubwordMerger...")
        self.subword_merger.apply_unigram_lm(num_merges)

    def apply_advanced_merging(self) -> None:
        logger.info("Starting advanced merging processes via SubwordMerger...")
        self.subword_merger.apply_advanced_merging()

    def _process_text(self, text: str) -> Counter[str]:
        return self.ngram_extractor.process_text(text, self.text_processor)

    def build_from_corpus(self, corpus: Iterable[str]) -> None:
        logger.info("Building vocabulary from corpus...")
        corpus_list = list(corpus)
        if not corpus_list:
            raise ValueError("Corpus is empty.")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._process_text, corpus_list))
        for counter in results:
            self.ngram_counts.update(counter)
        for ngram, count in self.ngram_counts.items():
            if count >= self.min_count:
                self._add_token(ngram, count)
                logger.info("Added n-gram %r with count %s", ngram, count)
        if self.use_advanced:
            self.apply_advanced_merging()
        logger.info("Vocabulary build complete.")

    def learn_in_real_time(self, text: str) -> None:
        logger.info("Real-time learning...")
        proc_text = self.text_processor.normalize(text)
        if self.mode == "word" and self.nlp:
            doc = self.text_processor.get_doc(proc_text)
            if doc:
                word_tokens = [t.text for t in doc if not t.is_space]
            else:
                word_tokens = proc_text.split()
            char_tokens = list(proc_text)
            for tokens, joiner in ((word_tokens, " "), (char_tokens, "")):
                token_count = len(tokens)
                for n in range(2, self.max_ngram + 1):
                    if token_count >= n:
                        for i in range(token_count - n + 1):
                            ngram = joiner.join(tokens[i : i + n])
                            self.ngram_counts[ngram] += 1
                            if (
                                self.ngram_counts[ngram] >= self.min_count
                                and ngram not in self.multi_token_vocab
                            ):
                                self._add_token(ngram, self.ngram_counts[ngram])
                                logger.info("Real-time added %r", ngram)
        else:
            tokens = list(proc_text)
            for n in range(2, self.max_ngram + 1):
                if len(tokens) >= n:
                    for i in range(len(tokens) - n + 1):
                        ngram = "".join(tokens[i : i + n])
                        self.ngram_counts[ngram] += 1
                        if (
                            self.ngram_counts[ngram] >= self.min_count
                            and ngram not in self.multi_token_vocab
                        ):
                            self._add_token(ngram, self.ngram_counts[ngram])
                            logger.info("Real-time added %r", ngram)
        self.executor.submit(self.corpus_manager.update, text)
        logger.info("Real-time learning update complete.")

    def generate_report(self) -> str:
        total_unique = len(self.ngram_counts)
        vocab_size = len(self.multi_token_vocab)
        report_lines = [
            "========== Vocabulary Report ==========",
            f"Total Unique N-grams: {total_unique}",
            f"Vocabulary Size: {vocab_size}",
            f"Average Frequency: {sum(self.multi_token_vocab.values()) / vocab_size if vocab_size else 0:.2f}",
            "Top 10 Tokens:",
        ]
        top_tokens = sorted(
            self.multi_token_vocab.items(), key=lambda x: (-x[1], x[0])
        )[:10]
        for token, count in top_tokens:
            report_lines.append(f"  {token!r}: {count}")
        report_lines.append("=======================================")
        return "\n".join(report_lines)

    def retokenize(self, text: str) -> List[str]:
        proc_text = self.text_processor.normalize(text)
        retok = []
        if self.mode == "word" and self.nlp:
            doc = self.text_processor.get_doc(proc_text)
            if not doc:
                tokens = proc_text.split()
            else:
                tokens = [token.text for token in doc]
            i = 0
            while i < len(tokens):
                found = None
                for size in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                    candidate = " ".join(tokens[i : i + size])
                    if candidate in self.multi_token_vocab:
                        found = candidate
                        break
                if found:
                    retok.append(found)
                    i += len(found.split())
                else:
                    fallback = self.trie.search_longest(tokens[i], 0)
                    if fallback:
                        retok.append(fallback)
                        i += len(fallback)
                    else:
                        retok.append(tokens[i])
                        i += 1
        else:
            i = 0
            while i < len(proc_text):
                token = self.trie.search_longest(proc_text, i)
                if token:
                    retok.append(token)
                    i += len(token)
                else:
                    retok.append(proc_text[i])
                    i += 1
        logger.info("Retokenization complete: %s", retok)
        return retok

    def save_vocab(self, filename: str) -> None:
        """
        Save the vocabulary state to a file in the proper tuple format:
        (multi_token_vocab, ngram_counts, token_metadata).
        This ensures the autosave file is always written in the correct format.
        """
        tmp_filename = filename + ".tmp"
        data = (self.multi_token_vocab, self.ngram_counts, self.token_metadata)
        try:
            with open(tmp_filename, "wb") as f:
                pickle.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_filename, filename)
            logger.info("Vocabulary saved to %s", filename)
        except Exception as e:
            logger.error("Error saving vocabulary: %s", e)
            raise

    def load_vocab(self, filename: str) -> None:
        try:
            with open(filename, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                # Old persistence format: {"vocab": vocab, "inverse_vocab": inverse_vocab}
                # For AdvancedVocabularyBuilder, we set ngram_counts and token_metadata to empty defaults.
                self.multi_token_vocab = loaded.get("vocab", {})
                self.ngram_counts = Counter()
                self.token_metadata = {}
            elif isinstance(loaded, tuple):
                if len(loaded) == 3:
                    self.multi_token_vocab, self.ngram_counts, self.token_metadata = (
                        loaded
                    )
                elif len(loaded) == 2:
                    self.multi_token_vocab, self.ngram_counts = loaded
                    self.token_metadata = {}
                else:
                    raise ValueError(
                        "Unexpected tuple length in loaded vocabulary file."
                    )
            else:
                raise ValueError("Unexpected format for loaded vocabulary.")
            logger.info("Vocabulary loaded from %s", filename)
            self.trie = Trie()
            for token in self.multi_token_vocab.keys():
                self.trie.insert(token)
        except Exception as e:
            logger.error("Error loading vocabulary: %s", e)
            raise

    def update_corpus(self, corpus: Iterable[str]) -> None:
        logger.info("Updating vocabulary with new corpus...")
        self.build_from_corpus(corpus)

    def get_vocabulary(self) -> Dict[str, int]:
        return dict(self.multi_token_vocab)

    def get_ngram_counts(self) -> Counter[str]:
        return self.ngram_counts

    def load_transformer_embedding_model(
        self, model_name: str = "distilbert-base-uncased"
    ) -> None:
        self.transformer_manager.load_model(model_name)
        logger.info("Transformer model loaded: %s", model_name)
        if self.corpus_manager.dynamic_corpus:
            texts = self.corpus_manager.dynamic_corpus
            inputs = self.transformer_manager.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            dataset = TensorDataset(torch.tensor(inputs["input_ids"]))
            self.executor.submit(self.transformer_manager.async_train, inputs, dataset)

    def update_dynamic_corpus(self, new_text: str) -> None:
        self.corpus_manager.update(new_text)

    def shutdown(self) -> None:
        """
        Shutdown the AdvancedVocabularyBuilder gracefully,
        ensuring autosave and executor shutdown.
        """
        logger.info("Shutting down AdvancedVocabularyBuilder...")
        self.auto_saver.stop()
        # Final autosave before shutdown (using lock already in save_vocab)
        try:
            self.save_vocab(str(self.auto_save_file))
            logger.info("Final autosave completed before shutdown.")
        except Exception as e:
            logger.error("Error during final autosave: %s", e)
        self.executor.shutdown(wait=True)
        logger.info("AdvancedVocabularyBuilder shut down.")


# -----------------------------------------------------------------------------
# CharTokenizer (basic character-level system that also supports plugins and persistence)
class CharTokenizer:
    DEFAULT_SPECIAL_TOKENS: Tuple[str, ...] = (
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[BOS]",
        "[EOS]",
        "[NUM]",
        "[FORMULA]",
        "[CODE]",
        "[EMOJI]",
        "[MEDIA]",
        "[AUDIO]",
        "[3D]",
    )
    UNICODE_STRATEGIES: Dict[str, Tuple[Tuple[int, int], ...]] = {
        "minimal": ((0x0000, 0x07FF),),
        "moderate": ((0x0000, 0xFFFF),),
        "extensive": ((0x0000, 0x10FFFF),),
        "technical": ((0x0000, 0x1FFFF), (0x20000, 0x2FFFF), (0xE0000, 0xEFFFF)),
        "mathematical": ((0x2000, 0x2BFF), (0x1D400, 0x1D7FF), (0x1EE00, 0x1EEFF)),
    }
    CATEGORY_PROFILES: Dict[str, Set[str]] = {
        "linguistic": {"L", "M", "N", "P", "S", "Z"},
        "technical": {"Sm", "Sc", "Sk", "So", "Nd", "No"},
        "formatting": {"Cc", "Cf", "Co", "Cn", "Zl", "Zp"},
        "symbolic": {"S", "So", "Sc", "Sk", "Sm"},
        "all": {"*"},
    }
    DEFAULT_CONTROL_CHARS: Tuple[int, ...] = (
        0x0000,
        0x0009,
        0x000A,
        0x000D,
        0x001B,
        0x007F,
        0x00A0,
        0x200B,
        0xFEFF,
    )

    def __init__(
        self,
        normalization_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC",
        special_tokens: Optional[Tuple[str, ...]] = None,
        unicode_strategy: Optional[str] = None,
        unicode_blocks: Optional[Iterable[Tuple[int, int]]] = None,
        category_profile: Optional[str] = None,
        technical_categories: Optional[Set[str]] = None,
        control_chars: Optional[Iterable[int]] = None,
        custom_chars: Optional[Iterable[str]] = None,
        sort_mode: Literal["unicode", "frequency", "custom"] = "unicode",
        dynamic_rebuild: bool = True,
        persistence_prefix: Optional[str] = None,
        **modes: Any,
    ):
        # Ensure special tokens are unique by filtering duplicates.
        # If special_tokens is provided and non-empty, we deduplicate it; otherwise, we fallback to DEFAULT_SPECIAL_TOKENS.
        tokens = (
            special_tokens
            if special_tokens is not None and special_tokens
            else self.DEFAULT_SPECIAL_TOKENS
        )
        unique_special_tokens = tuple(dict.fromkeys(tokens))
        self.config = TokenizerConfig(
            normalization_form=normalization_form,
            special_tokens=unique_special_tokens,
            unicode_strategy=unicode_strategy,
            unicode_blocks=unicode_blocks,
            category_profile=category_profile,
            technical_categories=technical_categories,
            control_chars=control_chars,
            custom_chars=custom_chars,
            sort_mode=sort_mode,
            dynamic_rebuild=dynamic_rebuild,
            persistence_prefix=persistence_prefix,
            modes=modes,
        )
        self._vocab_lock = threading.Lock()
        self.categories = self._resolve_categories(
            self.config.category_profile, self.config.technical_categories
        )
        self.unicode_blocks = self._resolve_unicode_coverage(
            self.config.unicode_strategy, self.config.unicode_blocks
        )
        self.control_chars = (
            set(self.config.control_chars)
            if self.config.control_chars
            else set(self.DEFAULT_CONTROL_CHARS)
        )
        self.custom_chars = (
            set(self.config.custom_chars) if self.config.custom_chars else set()
        )

        self.plugin_manager = PluginManager(self.config.modes)
        self._plugins = self.plugin_manager.plugins
        self.plugin_manager.attach_plugins(self)

        self.persistence_manager = (
            PersistenceManager(self.config.persistence_prefix)
            if self.config.persistence_prefix
            else None
        )

        # Here we use a simple vocabulary builder for basic char tokens.
        builder = SimpleVocabularyBuilder(
            self.config,
            self.unicode_blocks,
            self.categories,
            self.control_chars,
            self.custom_chars,
            self._plugins,
        )
        config_hash = self._get_config_hash()
        if self.persistence_manager:
            try:
                vocab_data = self.persistence_manager.load_vocabulary(config_hash)
            except Exception as e:
                logger.error("Error loading persistent vocabulary: %s", e)
                vocab_data = None

            if vocab_data:
                self.vocab = vocab_data["vocab"]
                self.inverse_vocab = vocab_data["inverse_vocab"]
                logger.info(f"{Fore.GREEN}Loaded vocabulary from persistence.")
            else:
                self.vocab, self.inverse_vocab = builder.build_vocabulary()
                self.persistence_manager.save_vocabulary(
                    config_hash, self.vocab, self.inverse_vocab
                )
        else:
            self.vocab, self.inverse_vocab = builder.build_vocabulary()

    def _resolve_unicode_coverage(
        self, strategy: Optional[str], custom: Optional[Iterable]
    ) -> Set[Tuple[int, int]]:
        blocks: Set[Tuple[int, int]] = set()
        if strategy:
            blocks.update(self.UNICODE_STRATEGIES.get(strategy, set()))
        # If custom is passed but is not a list/tuple, assume it's invalid and ignore it.
        if custom and not isinstance(custom, (list, tuple)):
            logger.warning(
                "Expected unicode_blocks to be an iterable of (int, int) tuples; ignoring value: %s",
                custom,
            )
            custom = None
        if custom:
            for item in custom:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and all(isinstance(x, int) for x in item)
                ):
                    blocks.add((item[0], item[1]))
                else:
                    logger.warning("Invalid unicode block format, skipping: %s", item)
        return blocks if blocks else set(self.UNICODE_STRATEGIES["extensive"])

    def _resolve_categories(
        self, profile: Optional[str], custom: Optional[Set]
    ) -> Set[str]:
        if profile == "all":
            return {"*"}
        categories = set()
        if profile:
            categories.update(self.CATEGORY_PROFILES.get(profile, set()))
        if custom:
            categories.update(custom)
        return categories or self.CATEGORY_PROFILES.get("technical", set())

    def _get_config_hash(self) -> str:
        config_str = (
            self.config.normalization_form,
            self.config.sort_mode,
            tuple(sorted(self.config.special_tokens or [])),
            tuple(sorted(self.unicode_blocks or [])),
            tuple(sorted(self.categories or [])),
            tuple(sorted(self.control_chars)),
            tuple(sorted(self.custom_chars)),
            self.config.modes,
        )
        return hashlib.md5(str(config_str).encode()).hexdigest()

    def _simple_vocabulary_builder(self):
        return SimpleVocabularyBuilder(
            self.config,
            self.unicode_blocks,
            self.categories,
            self.control_chars,
            self.custom_chars,
            self._plugins,
        )

    @lru_cache(maxsize=1024)
    def encode(
        self,
        text: str,
        normalization: Optional[Literal["NFC", "NFD", "NFKC", "NFKD"]] = None,
    ) -> List[int]:
        norm_form = (
            self.config.normalization_form if normalization is None else normalization
        )
        processed = unicodedata.normalize(norm_form, text)
        return [self.vocab.get(c, self.vocab.get("[UNK]", 0)) for c in processed]

    def decode(
        self,
        tokens: Iterable[int],
        normalization: Optional[Literal["NFC", "NFD", "NFKC", "NFKD"]] = None,
    ) -> str:
        decoded = "".join(self.inverse_vocab.get(t, "[UNK]") for t in tokens)
        return (
            unicodedata.normalize(normalization, decoded) if normalization else decoded
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        encoded = self.encode(text)
        return {
            "basic": {
                "length": len(text),
                "tokens": len(encoded),
                "ratio": len(encoded) / len(text) if text else 0,
            },
            "coverage": {
                "unknowns": encoded.count(self.vocab.get("[UNK]", 0)),
                "unique_chars": len(set(text)),
            },
            "unicode": {
                "normalized_form": self.config.normalization_form,
                "planes_used": len({(ord(c) >> 16) for c in text}),
            },
        }

    def generate_report(self) -> Dict[str, Any]:
        return {
            "configuration": {
                "normalization": self.config.normalization_form,
                "unicode_blocks": self.unicode_blocks,
                "categories": self.categories,
                "sort_mode": self.config.sort_mode,
            },
            "vocabulary": {
                "total_size": len(self.vocab),
                "special_tokens": len(self.config.special_tokens or []),
                "character_coverage": len(self.vocab)
                - len(self.config.special_tokens or []),
                "planes_covered": len(
                    {(ord(c) >> 16) for c in self.vocab if len(c) == 1}
                ),
            },
            "plugins": list(self._plugins.keys()),
        }

    def add_token(self, token: str) -> None:
        if token not in self.vocab:
            new_index = len(self.vocab)
            self.vocab[token] = new_index
            self.inverse_vocab[new_index] = token
            logger.info(f"Added token {token!r} with index {new_index}")
        else:
            logger.info(f"Token {token!r} already exists.")

    def reconfigure(self, **params) -> None:
        if not self.config.dynamic_rebuild:
            raise RuntimeError("Dynamic reconfiguration disabled")
        self.__dict__.update(params)
        self.plugin_manager = PluginManager(self.config.modes)
        self._plugins = self.plugin_manager.plugins
        self.plugin_manager.attach_plugins(self)
        self.categories = self._resolve_categories(
            self.config.category_profile, self.config.technical_categories
        )
        self.unicode_blocks = self._resolve_unicode_coverage(
            self.config.unicode_strategy, self.config.unicode_blocks
        )
        self.control_chars = (
            set(self.config.control_chars)
            if self.config.control_chars
            else set(self.DEFAULT_CONTROL_CHARS)
        )
        self.custom_chars = (
            set(self.config.custom_chars) if self.config.custom_chars else set()
        )
        builder = self._simple_vocabulary_builder()
        config_hash = self._get_config_hash()
        if self.persistence_manager:
            vocab_data = self.persistence_manager.load_vocabulary(config_hash)
            if vocab_data:
                self.vocab = vocab_data["vocab"]
                self.inverse_vocab = vocab_data["inverse_vocab"]
                logger.info(f"{Fore.GREEN}Loaded vocabulary from persistence.")
            else:
                self.vocab, self.inverse_vocab = builder.build_vocabulary()
                self.persistence_manager.save_vocabulary(
                    config_hash, self.vocab, self.inverse_vocab
                )
        else:
            self.vocab, self.inverse_vocab = builder.build_vocabulary()

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_plugins"] = {k: v.serialize() for k, v in self._plugins.items()}
        return state

    def __setstate__(self, state: Dict) -> None:
        plugins_state = state.pop("_plugins", {})
        self.__dict__.update(state)
        default_plugins = PluginManager(self.config.modes).plugins
        for name, serialized in plugins_state.items():
            if name in default_plugins and hasattr(
                default_plugins[name], "deserialize"
            ):
                default_plugins[name] = default_plugins[name].deserialize(serialized)
            else:
                default_plugins[name] = serialized
        self._plugins = default_plugins
        PluginManager(self.config.modes).attach_plugins(self)


# -----------------------------------------------------------------------------
# A very simple vocabulary builder for fallback (builds from Unicode sources, custom tokens, and plugins)
class SimpleVocabularyBuilder:
    def __init__(
        self,
        config: TokenizerConfig,
        unicode_blocks: Set[Tuple[int, int]],
        categories: Set[str],
        control_chars: Set[int],
        custom_chars: Set[str],
        plugins: Dict[str, BasePlugin],
    ):
        self.config = config
        self.unicode_blocks = unicode_blocks
        self.categories = categories
        self.control_chars = control_chars
        self.custom_chars = custom_chars
        self.plugins = plugins

    def validate_configuration(self) -> None:
        if len(self.config.special_tokens or []) != len(
            set(self.config.special_tokens or [])
        ):
            raise ValueError("Special tokens must be unique")
        for char in self.custom_chars:
            if not (isinstance(char, str) and len(char) == 1):
                raise ValueError("Custom characters must be single-character strings")

    def add_unicode_blocks(self, chars: Set[str]) -> None:
        for start, end in self.unicode_blocks:
            if (
                start > end
                or not (0 <= start <= 0x10FFFF)
                or not (0 <= end <= 0x10FFFF)
            ):
                raise ValueError(f"Invalid Unicode range: {start}-{end}")
            chars.update({chr(c) for c in range(start, end + 1)})

    def add_unicode_categories(self, chars: Set[str]) -> None:
        # If the categories include the wildcard "*", do nothing—
        # we assume that means "all" and nothing extra needs to be added.
        if "*" in self.categories:
            return
        # For each category, compute the allowed intervals
        for cat in self.categories:
            intervals = UnicodeUtils.compute_intervals_for_category(cat)
            for start, end in intervals:
                # Instead of adding every code point in the interval,
                # add only those characters which are printable.
                chars.update(
                    {chr(c) for c in range(start, end + 1) if chr(c).isprintable()}
                )

    def add_control_chars(self, chars: Set[str]) -> None:
        chars.update({chr(c) for c in self.control_chars})

    def sort_vocabulary(self, chars: Iterable[str]) -> List[str]:
        if self.config.sort_mode == "unicode":
            # Sort tokens in increasing order (by Unicode code point)
            return sorted(chars, key=lambda c: ord(c))
        elif self.config.sort_mode == "frequency":
            # Minimal placeholder: sort in descending order of Unicode code point.
            # (In a full implementation, you would sort by token frequency.)
            return sorted(chars, key=lambda c: -ord(c))
        return list(chars)

    def build_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        self.validate_configuration()
        vocab = {t: i for i, t in enumerate(self.config.special_tokens or [])}
        chars: Set[str] = set()
        self.add_unicode_blocks(chars)
        self.add_unicode_categories(chars)
        self.add_control_chars(chars)
        chars.update(self.custom_chars)
        for plugin in self.plugins.values():
            try:
                chars.update(plugin.get_chars())
            except Exception as e:
                logger.error(f"Plugin error in get_chars: {e}")
        ordered_chars = self.sort_vocabulary(chars)
        for idx, char in enumerate(ordered_chars, start=len(vocab)):
            vocab[char] = idx
        inverse_vocab = {v: k for k, v in vocab.items()}
        return vocab, inverse_vocab


# -----------------------------------------------------------------------------
# Dynamic Tokenizer – combining character tokenization with advanced dynamic vocabulary learning
class DynamicTokenizer(CharTokenizer):
    def __init__(
        self,
        normalization_form,
        unicode_strategy,
        category_profile,
        sort_mode,
        dynamic_rebuild,
        persistence_prefix,
        **kwargs,
    ):
        # Fix: pass parameters by name so they fall into the intended slots.
        super().__init__(
            normalization_form=normalization_form,
            special_tokens=None,  # use default special tokens
            unicode_strategy=unicode_strategy,
            unicode_blocks=None,  # no custom unicode blocks provided
            category_profile=category_profile,
            technical_categories=None,  # no technical categories
            control_chars=None,  # use default control characters
            custom_chars=None,  # no custom characters
            sort_mode=sort_mode,
            dynamic_rebuild=dynamic_rebuild,
            persistence_prefix=persistence_prefix,
            **kwargs,
        )
        self.advanced_vocab_builder = AdvancedVocabularyBuilder(self)
        # Start a background continuous learning thread which periodically performs a comprehensive vocabulary update.
        self._shutdown_event = threading.Event()
        self.background_learning_thread = threading.Thread(
            target=self._continuous_learning_loop, daemon=True
        )
        self.background_learning_thread.start()

    def _continuous_learning_loop(self) -> None:
        """
        Continuously run comprehensive vocabulary updates in the background every 5 minutes.
        """
        while not self._shutdown_event.is_set():
            time.sleep(300)  # sleep for 5 minutes
            try:
                logger.info(
                    "Background continuous learning: initiating comprehensive vocabulary update..."
                )
                self.advanced_vocab_builder.learn_and_update()
            except Exception as e:
                logger.error("Error in background continuous learning: %s", e)

    def learn(self, text: str) -> None:
        # Example encoding and token learning process.
        token_ids = self.encode(text)
        for char in text:
            if char not in self.vocab:
                self.add_token(char)
        # Use the advanced vocabulary builder to perform a comprehensive update
        # of multi-token n-grams and contextual tokens.
        self.advanced_vocab_builder.learn_and_update()
        if self.persistence_manager:
            config_hash = self._get_config_hash()
            try:
                self.persistence_manager.save_vocabulary(
                    config_hash, self.vocab, self.inverse_vocab
                )
                logger.info("Persisted vocabulary after learning.")
            except Exception as e:
                logger.error("Error persisting vocabulary: %s", e)

    def learn_in_background(self, text: str) -> None:
        # Launch learning in a background daemon thread without blocking the main thread.
        thread = threading.Thread(target=self.learn, args=(text,), daemon=True)
        thread.start()

    def shutdown(self) -> None:
        """
        Shutdown the dynamic tokenizer: stop background tasks and gracefully shut down the advanced vocabulary builder.
        """
        logger.info("Shutting down DynamicTokenizer...")
        self._shutdown_event.set()
        if self.background_learning_thread.is_alive():
            self.background_learning_thread.join(timeout=5)
        self.advanced_vocab_builder.shutdown()
        logger.info("DynamicTokenizer shut down.")


# -----------------------------------------------------------------------------
# Demonstration Routine
def demo_intelligent_tokenizer() -> None:
    init(autoreset=True)
    logger.info(
        f"\n{Back.BLUE}{Fore.WHITE}=== Intelligent Adaptive Tokenizer Demo ==={Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}Initializing dynamic intelligent tokenizer with advanced vocabulary building..."
    )
    tokenizer = DynamicTokenizer(
        normalization_form="NFC",
        unicode_strategy="extensive",
        category_profile="all",
        sort_mode="unicode",
        dynamic_rebuild=True,
        persistence_prefix="intelligent_tokenizer",  # Enable persistence
    )
    sample_text = "A sample evolving text with new tokens: 🚀✨"
    # Kick off an initial background learning process with a sample text.
    tokenizer.learn_in_background(sample_text)
    logger.info(
        f"{Fore.GREEN}Dynamic Intelligent Tokenizer ready! Vocabulary size: {len(tokenizer.vocab)} tokens"
    )
    try:
        while True:
            logger.info(
                f"\n{Back.MAGENTA}{Fore.WHITE}=== Demo Menu ==={Style.RESET_ALL}"
            )
            logger.info("1. Test your own text")
            logger.info("2. Multilingual example")
            logger.info("3. Emoji/Technical example")
            logger.info("4. System report")
            logger.info("5. Learn new text dynamically")
            logger.info("6. Exit")
            logger.info("7. Learn and Update (comprehensive vocabulary update)")
            choice = input(f"{Fore.WHITE}Choose an option (1-7): ").strip()
            if choice == "1":
                text = input("Enter text: ").strip()
                if not text:
                    logger.info("Please enter some text!")
                    continue
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded)
                analysis = tokenizer.analyze(text)
                logger.info(f"Encoded tokens: {encoded}")
                logger.info(f"Decoded text: {decoded}")
                logger.info(f"Analysis: {analysis}")
            elif choice == "2":
                sample = "日本語 текст! 123 👨💻 + ∑x² = 42 🚀"
                logger.info(f"Multilingual sample: {sample}")
                encoded = tokenizer.encode(sample)
                decoded = tokenizer.decode(encoded)
                logger.info(f"Encoded: {encoded}")
                logger.info(f"Decoded: {decoded}")
            elif choice == "3":
                sample = "🔥🐉 Schrödinger's Cat: ⚛️📈 ∑x² ⇒ ∞ 😱💥"
                logger.info(f"Emoji/Technical sample: {sample}")
                encoded = tokenizer.encode(sample)
                decoded = tokenizer.decode(encoded)
                logger.info(f"Encoded: {encoded}")
                logger.info(f"Decoded: {decoded}")
            elif choice == "4":
                report = tokenizer.generate_report()
                logger.info(f"System Report: {report}")
            elif choice == "5":
                learn_text = input("Enter text to learn dynamically: ").strip()
                if not learn_text:
                    logger.info("Please enter some text!")
                    continue
                tokenizer.learn_in_background(learn_text)
                logger.info(f"Vocabulary updated! New size: {len(tokenizer.vocab)}")
            elif choice == "7":
                logger.info("Initiating comprehensive learn and update process...")
                tokenizer.advanced_vocab_builder.learn_and_update()
                logger.info(
                    f"Learn and update process complete. Current vocabulary size: {len(tokenizer.vocab)}"
                )
            elif choice == "6":
                logger.info("Exiting demo...")
                break
            else:
                logger.info("Invalid choice. Try again.")
    except KeyboardInterrupt:
        logger.info("Interrupt received; shutting down...")
    finally:
        tokenizer.shutdown()
        logger.info("Demo complete.")


if __name__ == "__main__":
    demo_intelligent_tokenizer()
