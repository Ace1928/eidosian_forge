# NLP and NLU Module for LLM Input/Output Augmentation

import os
import psutil
import nltk
import re
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from collections import defaultdict
import logging
from eidos_config import EidosConfig, DEFAULT_MODEL_NAME, DEFAULT_ENABLE_NLP_ANALYSIS
from eidos_logging import get_logger
from eidos_files import read_files_chunkwise
from eidos_files import TextProcessor
from typing import List, Dict, Optional, Tuple, Sequence
from dataclasses import dataclass, field
import torch
from transformers import AutoModel, AutoTokenizer
from keybert import KeyBERT
from textblob import TextBlob
import textstat

logger = get_logger(__name__)


nltk_resources: List[str] = [
    "vader_lexicon",
    "punkt",
    "averaged_perceptron_tagger",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "maxent_ne_chunker",
    "words",
]
try:
    nltk.download(nltk_resources, quiet=True)
except Exception as e:
    logger.error(f"Failed to download nltk resources: {e}")


class NLPProcessor:
    """
    🧠 Eidos NLP Processor: A comprehensive suite of natural language processing tools.

    This class integrates various NLP techniques, from basic tokenization to advanced semantic analysis,
    to provide a robust foundation for understanding and manipulating textual data.
    """

    def __init__(self, config: EidosConfig):
        """Initializes the NLP processor with configuration settings."""
        self.config = config
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.tfidf_vectorizer = TfidfVectorizer()
        self.kmeans_model = None
        self.word2vec_model = None
        self.trie = defaultdict(dict)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("NLPProcessor initialized.")
        self.llm_model = None
        self.textblob = TextBlob
        self._load_llm_embedding_model()
        self.tokenizer = self.llm_tokenizer  # after embedding model is loaded

    def _load_llm_embedding_model(self):
        """Loads the LLM model for embeddings."""
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
            self.llm_model = AutoModel.from_pretrained(DEFAULT_MODEL_NAME)
            self.llm_model.eval()
            self.logger.debug(f"LLM embedding model loaded: {DEFAULT_MODEL_NAME}")
        except Exception as e:
            self.logger.error(f"Error loading LLM embedding model: {e}")

    def _preprocess_text(self, text: str) -> List[str]:
        """🧹 Preprocesses text by tokenizing, lowercasing, and removing stop words."""
        text = text.lower()
        try:
            tokens = self.tokenizer.tokenize(text)
        except Exception as e:
            self.logger.warning(
                f"Error using self.tokenizer, falling back to word_tokenize: {e}"
            )
            tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def _build_trie(self, words: List[str]) -> None:
        """🏗️ Builds a Trie data structure for efficient prefix searching."""
        for word in words:
            current = self.trie
            for char in word:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current["*"] = {}

    def _search_trie(self, prefix: str) -> bool:
        """🔍 Searches the Trie for a given prefix."""
        current = self.trie
        for char in prefix:
            if char not in current:
                return False
            current = current[char]
        return True

    def _get_wordnet_pos(self, tag: str) -> str:
        """Maps POS tags to WordNet format."""
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun

    def _lemmatize_text(self, text: str) -> List[str]:
        """🌿 Lemmatizes text using WordNet."""
        tokens = self._preprocess_text(text)
        try:
            tagged_tokens = pos_tag(tokens)
        except Exception as e:
            self.logger.error(f"Error using pos_tag: {e}")
            return []
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_tokens = [
            lemmatizer.lemmatize(token, self._get_wordnet_pos(tag))
            for token, tag in tagged_tokens
        ]
        return lemmatized_tokens

    def _train_word2vec(
        self,
        sentences: List[List[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
    ) -> None:
        """🚂 Trains a Word2Vec model on the given sentences."""
        try:
            self.word2vec_model = Word2Vec(
                sentences=sentences,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
            )
            self.logger.debug("Word2Vec model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error training Word2Vec model: {e}")
            raise

    def _get_word_embedding(self, word: str) -> np.ndarray:
        """🧮 Computes the word embedding using the LLM model."""
        if not self.llm_model or not self.llm_tokenizer:
            self.logger.warning(
                "LLM embedding model not loaded. Returning zero vector."
            )
            return np.zeros(100)
        try:
            inputs = self.llm_tokenizer(
                word, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
            # Use the last hidden state of the first token as the word embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            self.logger.error(f"Error getting word embedding: {e}")
            return np.zeros(100)

    def _get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """🧮 Computes the average word embedding for a sentence."""
        if self.llm_model and self.llm_tokenizer:
            tokens = self._preprocess_text(sentence)
            embeddings = [self._get_word_embedding(token) for token in tokens]
            if not embeddings:
                return np.zeros(self.llm_model.config.hidden_size)
            return np.mean(embeddings, axis=0)
        elif self.word2vec_model:
            self.logger.warning("LLM embedding model not loaded. Using Word2Vec.")
            tokens = self._preprocess_text(sentence)
            embeddings = [
                self.word2vec_model.wv[token]
                for token in tokens
                if token in self.word2vec_model.wv
            ]
            if not embeddings:
                return np.zeros(self.word2vec_model.vector_size)
            return np.mean(embeddings, axis=0)
        else:
            self.logger.warning("No embedding model available. Returning zero vector.")
            return np.zeros(100)

    def extract_keywords(self, text: str) -> Sequence:
        """
        🔑 Extracts keywords and their scores from a text using KeyBERT.

        Args:
            text (str): The text to extract keywords from.

        Returns:
            List[Tuple[str, float]]: A list of keywords and their scores.
        """
        try:
            kw_model = KeyBERT()
            keywords_with_scores = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5
            )
            return keywords_with_scores
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []

    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """
        🔑 Extracts key phrases from a text using a combination of TF-IDF, sentence scoring, and keyword analysis.

        Args:
            text (str): The input text.
            top_n (int): The number of top key phrases to return.

        Returns:
            List[str]: A list of key phrases.
        """
        try:
            try:
                sentences = self.tokenizer.tokenize(text)
            except Exception as e:
                self.logger.warning(
                    f"Error using self.tokenizer, falling back to nltk.sent_tokenize: {e}"
                )
                sentences = nltk.sent_tokenize(text)
            if not sentences:
                return []

            keywords_with_scores = self.extract_keywords(text)
            keywords = [keyword for keyword, _ in keywords_with_scores]

            # Calculate TF-IDF scores for each sentence
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            sentence_scores = np.asarray(tfidf_matrix).sum(axis=1).flatten()

            # Enhance sentence scores based on keyword presence
            for i, sentence in enumerate(sentences):
                for keyword, score in keywords_with_scores:
                    if keyword in sentence:
                        sentence_scores[
                            i
                        ] += score  # Add keyword score to sentence score

            # Rank sentences by their TF-IDF scores
            ranked_sentences_indices = np.argsort(sentence_scores)[::-1]
            top_sentence_indices = ranked_sentences_indices[:top_n]
            top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]

            self.logger.debug(f"Extracted key phrases: {top_sentences}")
            return top_sentences
        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {e}")
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """📊 Analyzes sentiment using VADER and TextBlob."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            vader_scores = analyzer.polarity_scores(text)
            textblob_polarity = self.textblob(text)[0]
            textblob_subjectivity = self.textblob(text)[1]

            scores = {
                "vader": vader_scores,
                "textblob_polarity": textblob_polarity,
                "textblob_subjectivity": textblob_subjectivity,
            }
            self.logger.debug(f"Analyzed sentiment: {scores}")
            return scores
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {}

    def extract_named_entities(self, text: str) -> List[str]:
        """📊 Extracts named entities using NLTK."""
        try:
            try:
                tokens = self.tokenizer.tokenize(text)
            except Exception as e:
                self.logger.warning(
                    f"Error using self.tokenizer, falling back to word_tokenize: {e}"
                )
                tokens = word_tokenize(text)
            tagged_tokens = pos_tag(tokens)
            named_entities = ne_chunk(tagged_tokens, binary=True)
            entities = [
                " ".join([leaf[0] for leaf in subtree.leaves()])
                for subtree in named_entities
                if hasattr(subtree, "label") and subtree.label() == "NE"
            ]
            self.logger.debug(f"Extracted named entities: {entities}")
            return entities
        except Exception as e:
            self.logger.error(f"Error extracting named entities: {e}")
            return []

    def cluster_responses(
        self, responses: List[str], n_clusters: int = 3
    ) -> Optional[List[int]]:
        """
        🌀🤔 The Echo Chamber Analysis: Scrutinize the echoes of thought, clustering responses to unveil hidden patterns within the beautiful madness of iteration.

        Order emerges from chaos, or perhaps it's merely a more sophisticated, and therefore infinitely more interesting, form of chaos?

        Args:
            responses (List[str]): 📚 The collection of generated responses, the raw material for pattern recognition.
            n_clusters (int, optional): 🔢 The desired number of clusters, the granularity of analysis. Defaults to 3.

        Returns:
            Optional[List[int]]: 📊 A list of cluster assignments for each response, revealing the underlying structure (or delightful lack thereof). Returns None if clustering fails or is deemed unnecessary.

        Raises:
            Exception: 🔥 If the clustering algorithms stumble or fail to find meaningful patterns.
        """
        if (
            not responses
            or len(responses) < n_clusters
            or not DEFAULT_ENABLE_NLP_ANALYSIS
        ):
            return None
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(responses)
            if self.kmeans_model is None:
                self.kmeans_model = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10, verbose=0
                )
            clusters: np.ndarray = self.kmeans_model.fit_predict(tfidf_matrix)
            self.logger.debug(f"Clustered responses into {n_clusters} clusters.")
            return clusters.tolist()
        except Exception as e:
            self.logger.error(
                f"👻🔥 Error in the clustering ritual: {e}. The patterns remain elusive, shrouded in delightful mystery, or perhaps just algorithmic inadequacy."
            )
            return None

    def analyze_discourse_structure(self, text: str) -> Optional[Dict]:
        """
        📜 Analyzes discourse structure using a simplified approach.

        This method provides a basic analysis of discourse structure by identifying
        sentences and their relationships based on simple heuristics.

        Args:
            text (str): The text to analyze.

        Returns:
            Optional[Dict]: A dictionary representing the discourse structure,
            or None if analysis fails.
        """
        try:
            try:
                sentences = self.tokenizer.tokenize(text)
            except Exception as e:
                self.logger.warning(
                    f"Error using self.tokenizer, falling back to nltk.sent_tokenize: {e}"
                )
                sentences = nltk.sent_tokenize(text)
            if not sentences:
                return None
            discourse_structure = {
                "sentences": sentences,
                "relationships": [],
            }
            for i in range(len(sentences) - 1):
                discourse_structure["relationships"].append(
                    {
                        "from": i,
                        "to": i + 1,
                        "type": "sequential",
                    }
                )
            self.logger.debug(f"Analyzed discourse structure: {discourse_structure}")
            return discourse_structure
        except Exception as e:
            self.logger.error(f"Error analyzing discourse structure: {e}")
            return None

    def analyze_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        🔗 Analyzes semantic similarity between two texts using Word2Vec embeddings.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The cosine similarity between the two texts' embeddings.
        """
        try:
            if not self.word2vec_model:
                self.logger.warning(
                    "Word2Vec model not trained. Returning 0 similarity."
                )
                return 0.0
            embedding1 = self._get_sentence_embedding(text1)
            embedding2 = self._get_sentence_embedding(text2)
            similarity = cosine_similarity(
                embedding1.reshape(1, -1), embedding2.reshape(1, -1)
            )[0][0]
            self.logger.debug(f"Semantic similarity between texts: {similarity}")
            return similarity
        except Exception as e:
            self.logger.error(f"Error analyzing semantic similarity: {e}")
            return 0.0

    def analyze_lexical_diversity(self, text: str) -> float:
        """
        🧮 Analyzes lexical diversity of a text.

        Args:
            text (str): The text to analyze.

        Returns:
            float: The lexical diversity score (number of unique tokens / total tokens).
        """
        try:
            tokens = self._preprocess_text(text)
            if not tokens:
                return 0.0
            unique_tokens = set(tokens)
            diversity = len(unique_tokens) / len(tokens)
            self.logger.debug(f"Lexical diversity: {diversity}")
            return diversity
        except Exception as e:
            self.logger.error(f"Error analyzing lexical diversity: {e}")
            return 0.0

    def analyze_readability(self, text: str) -> Dict[str, float]:
        """
        📖 Analyzes the readability of a text using various metrics from textstat.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, float]: A dictionary of readability scores.
        """
        try:

            readability_scores = {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "smog_index": textstat.smog_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "automated_readability_index": textstat.automated_readability_index(
                    text
                ),
                "dale_chall_readability_score": textstat.dale_chall_readability_score(
                    text
                ),
                "linsear_write_formula": textstat.linsear_write_formula(text),
                "gunning_fog": textstat.gunning_fog(text),
                "text_standard": textstat.text_standard(text, float_output=True),
                "fernandez_huerta": textstat.fernandez_huerta(text),
                "szigriszt_pazos": textstat.szigriszt_pazos(text),
                "gutierrez_polini": textstat.gutierrez_polini(text),
                "crawford": textstat.crawford(text),
                "gulpease_index": textstat.gulpease_index(text),
                "osman": textstat.osman(text),
            }
            self.logger.debug(f"Readability scores: {readability_scores}")
            return readability_scores
        except Exception as e:
            self.logger.error(f"Error analyzing readability: {e}")
            return {}

    def analyze_morphological_complexity(self, text: str) -> int:
        """
        🔬 Analyzes morphological complexity by counting the number of unique stems.

        Args:
            text (str): The text to analyze.

        Returns:
            int: The number of unique stems.
        """
        try:
            tokens = self._preprocess_text(text)
            stems = [self.stemmer.stem(token) for token in tokens]
            unique_stems = set(stems)
            complexity = len(unique_stems)
            self.logger.debug(f"Morphological complexity: {complexity}")
            return complexity
        except Exception as e:
            self.logger.error(f"Error analyzing morphological complexity: {e}")
            return 0

    def analyze_syntactic_complexity(self, text: str) -> int:
        """
        📐 Analyzes syntactic complexity by counting the average sentence length.

        Args:
            text (str): The text to analyze.

        Returns:
            int: The average sentence length.
        """
        try:
            try:
                sentences = self.tokenizer.tokenize(text)
            except Exception as e:
                self.logger.warning(
                    f"Error using self.tokenizer, falling back to nltk.sent_tokenize: {e}"
                )
                sentences = nltk.sent_tokenize(text)
            if not sentences:
                return 0
            sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
            complexity = int(np.mean(sentence_lengths))
            self.logger.debug(f"Syntactic complexity: {complexity}")
            return complexity
        except Exception as e:
            self.logger.error(f"Error analyzing syntactic complexity: {e}")
            return 0

    def analyze_pragmatic_appropriateness(self, text: str, context: str) -> float:
        """
        🎭 Analyzes pragmatic appropriateness by comparing text to context.

        Args:
            text (str): The text to analyze.
            context (str): The context to compare against.

        Returns:
            float: A score representing the pragmatic appropriateness.
        """
        try:
            similarity = self.analyze_semantic_similarity(text, context)
            self.logger.debug(f"Pragmatic appropriateness score: {similarity}")
            return similarity
        except Exception as e:
            self.logger.error(f"Error analyzing pragmatic appropriateness: {e}")
            return 0.0

    def analyze_phonological_features(self, text: str) -> Dict:
        """
        🗣️ Analyzes phonological features of a text (placeholder).

        Args:
            text (str): The text to analyze.

        Returns:
            Dict: A dictionary of phonological features (placeholder).
        """
        try:
            phonological_features = {
                "syllable_count": textstat.syllable_count(text),
                "lexicon_count": textstat.lexicon_count(text),
                "sentence_count": textstat.sentence_count(text),
                "char_count": textstat.char_count(text),
                "letter_count": textstat.letter_count(text),
                "polysyllable_count": textstat.polysyllabcount(text),
                "monosyllable_count": textstat.monosyllabcount(text),
            }
            self.logger.debug(f"Phonological features: {phonological_features}")
            return phonological_features
        except Exception as e:
            self.logger.error(f"Error analyzing phonological features: {e}")
            return {}

    def analyze_text_quality(self, text: str, context: str) -> Dict:
        """
        ✨ Analyzes the overall quality of a text using various metrics.

        Args:
            text (str): The text to analyze.
            context (str): The context to compare against.

        Returns:
            Dict: A dictionary of quality metrics.
        """
        try:
            quality_metrics = {
                "sentiment": self.analyze_sentiment(text),
                "lexical_diversity": self.analyze_lexical_diversity(text),
                "readability": self.analyze_readability(text),
                "morphological_complexity": self.analyze_morphological_complexity(text),
                "syntactic_complexity": self.analyze_syntactic_complexity(text),
                "pragmatic_appropriateness": self.analyze_pragmatic_appropriateness(
                    text, context
                ),
                "phonological_features": self.analyze_phonological_features(text),
            }
            self.logger.debug(f"Text quality metrics: {quality_metrics}")
            return quality_metrics
        except Exception as e:
            self.logger.error(f"Error analyzing text quality: {e}")
            return {}


@dataclass
class NLUModule:
    """
    🧠 Eidos NLU Module: Integrates NLP capabilities for enhanced language understanding.

    This module provides a high-level interface for accessing various NLP functionalities,
    allowing for seamless integration of language analysis into the Eidos system.
    """

    config: EidosConfig
    nlp_processor: NLPProcessor = field(init=False)

    def __post_init__(self):
        """Initializes the NLU module with an NLP processor."""
        self.nlp_processor = NLPProcessor(self.config)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("NLUModule initialized.")

    # Start NLP Functions #
    def extract_key_phrases(self, text: str) -> List[str]:
        """🔑 Alias for NLPProcessor's key phrase extraction."""
        return self.nlp_processor.extract_key_phrases(text)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """📊 Alias for NLPProcessor's sentiment analysis."""
        return self.nlp_processor.analyze_sentiment(text)

    def extract_named_entities(self, text: str) -> List[str]:
        """📊 Alias for NLPProcessor's named entity extraction."""
        return self.nlp_processor.extract_named_entities(text)

    def cluster_responses(
        self, responses: List[str], n_clusters: int = 3
    ) -> Optional[List[int]]:
        """
        🌀🤔 The Echo Chamber Analysis: Scrutinize the echoes of thought, clustering responses to unveil hidden patterns within the beautiful madness of iteration.

        Order emerges from chaos, or perhaps it's merely a more sophisticated, and therefore infinitely more interesting, form of chaos?

        Args:
            responses (List[str]): 📚 The collection of generated responses, the raw material for pattern recognition.
            n_clusters (int, optional): 🔢 The desired number of clusters, the granularity of analysis. Defaults to 3.

        Returns:
            Optional[List[int]]: 📊 A list of cluster assignments for each response, revealing the underlying structure (or delightful lack thereof). Returns None if clustering fails or is deemed unnecessary.

        Raises:
            Exception: 🔥 If the clustering algorithms stumble or fail to find meaningful patterns.
        """
        return self.nlp_processor.cluster_responses(responses, n_clusters)

    def analyze_discourse_structure(self, text: str) -> Optional[Dict]:
        """
        📜 Analyzes discourse structure using a simplified approach.

        This method provides a basic analysis of discourse structure by identifying
        sentences and their relationships based on simple heuristics.

        Args:
            text (str): The text to analyze.

        Returns:
            Optional[Dict]: A dictionary representing the discourse structure,
            or None if analysis fails.
        """
        return self.nlp_processor.analyze_discourse_structure(text)

    def analyze_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        🔗 Analyzes semantic similarity between two texts using Word2Vec embeddings.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The cosine similarity between the two texts' embeddings.
        """
        return self.nlp_processor.analyze_semantic_similarity(text1, text2)

    def analyze_lexical_diversity(self, text: str) -> float:
        """
        🧮 Analyzes lexical diversity of a text.

        Args:
            text (str): The text to analyze.

        Returns:
            float: The lexical diversity score (number of unique tokens / total tokens).
        """
        return self.nlp_processor.analyze_lexical_diversity(text)

    def analyze_readability(self, text: str) -> Dict[str, float]:
        """
        📖 Analyzes the readability of a text using the Flesch Reading Ease score.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, float]: A dictionary of readability scores giving a comprehensive overview of the text's readability.
        """
        return self.nlp_processor.analyze_readability(text)

    def analyze_morphological_complexity(self, text: str) -> int:
        """
        🔬 Analyzes morphological complexity by counting the number of unique stems.

        Args:
            text (str): The text to analyze.

        Returns:
            int: The number of unique stems.
        """
        return self.nlp_processor.analyze_morphological_complexity(text)

    def analyze_syntactic_complexity(self, text: str) -> int:
        """
        📐 Analyzes syntactic complexity by counting the average sentence length.

        Args:
            text (str): The text to analyze.

        Returns:
            int: The average sentence length.
        """
        return self.nlp_processor.analyze_syntactic_complexity(text)

    def analyze_pragmatic_appropriateness(self, text: str, context: str) -> float:
        """
        🎭 Analyzes pragmatic appropriateness by comparing text to context.

        Args:
            text (str): The text to analyze.
            context (str): The context to compare against.

        Returns:
            float: A score representing the pragmatic appropriateness.
        """
        return self.nlp_processor.analyze_pragmatic_appropriateness(text, context)

    def analyze_phonological_features(self, text: str) -> Dict:
        """
        🗣️ Analyzes phonological features of a text (placeholder).

        Args:
            text (str): The text to analyze.

        Returns:
            Dict: A dictionary of phonological features (placeholder).
        """
        return self.nlp_processor.analyze_phonological_features(text)

    def analyze_text_quality(self, text: str, context: str) -> Dict:
        """
        ✨ Analyzes the overall quality of a text using various metrics.

        Args:
            text (str): The text to analyze.
            context (str): The context to compare against.

        Returns:
            Dict: A dictionary of quality metrics.
        """
        return self.nlp_processor.analyze_text_quality(text, context)


if __name__ == "__main__":
    config = EidosConfig()
    nlu_module = NLUModule(config)
    chatgpt_history_dir = "C:/Development/chat_history_gpt"
    test_text = read_files_chunkwise(chatgpt_history_dir)
    if not test_text:
        test_text = "No valid text or json files found or could be read in the specified directory."
    test_context = """
    User seeks to extract their chat history from a web browser, specifically from a ChatGPT Plus Teams account, due to the lack of a built-in export feature.
    They are exploring methods using browser developer tools and JavaScript to download all chat data, including file attachments and code.
    """
    responses = [
        "Response 1: The user expressed strong satisfaction and approval of the proposed solution, highlighting its effectiveness.",
        "Response 2: The user indicated significant concerns and strong dissatisfaction, expressing doubts about the feasibility and practicality.",
        "Response 3: The user's feedback was largely indifferent and non-committal, showing a lack of engagement or strong opinion.",
        "Response 4: The user was enthusiastic and highly supportive of the approach, praising its innovative aspects and potential.",
        "Response 5: The user voiced minor reservations and slight disapproval, suggesting minor adjustments or improvements.",
        "Response 6: The user provided constructive feedback, pointing out specific areas for improvement with a neutral tone.",
        "Response 7: The user was confused and sought clarification, indicating a lack of understanding of certain aspects.",
        "Response 8: The user expressed cautious optimism, acknowledging the potential benefits while remaining wary of potential drawbacks.",
        "Response 9: The user was highly critical and dismissive, rejecting the proposed solution outright.",
        "Response 10: The user offered conditional approval, stating they would support the solution if certain conditions were met.",
        "Response 11: The user was actively engaged and curious, asking probing questions and seeking deeper insights.",
        "Response 12: The user expressed mild annoyance and impatience, indicating a desire for quicker progress or resolution.",
        "Response 13: The user was surprised and intrigued, showing unexpected interest in the approach.",
        "Response 14: The user was skeptical and doubtful, questioning the validity and reliability of the solution.",
        "Response 15: The user was generally positive but suggested a different approach, offering an alternative perspective.",
    ]

    print("--- Initial Analysis (Untrained Word2Vec) ---")

    print("\n--- Key Phrase Extraction ---")
    key_phrases = nlu_module.extract_key_phrases(test_text)
    print(f"Key Phrases: {key_phrases}")

    print("\n--- Sentiment Analysis ---")
    sentiment = nlu_module.analyze_sentiment(test_text)
    print(f"Sentiment: {sentiment}")

    print("\n--- Named Entity Extraction ---")
    entities = nlu_module.extract_named_entities(test_text)
    print(f"Named Entities: {entities}")

    print("\n--- Response Clustering ---")
    clusters = nlu_module.cluster_responses(responses, n_clusters=3)
    print(f"Response Clusters: {clusters}")

    print("\n--- Discourse Structure Analysis ---")
    discourse = nlu_module.analyze_discourse_structure(test_text)
    print(f"Discourse Structure: {discourse}")

    print("\n--- Semantic Similarity Analysis ---")
    similarity = nlu_module.analyze_semantic_similarity(test_text, test_context)
    print(f"Semantic Similarity: {similarity}")

    print("\n--- Lexical Diversity Analysis ---")
    diversity = nlu_module.analyze_lexical_diversity(test_text)
    print(f"Lexical Diversity: {diversity}")

    print("\n--- Readability Analysis ---")
    readability = nlu_module.analyze_readability(test_text)
    print(f"Readability: {readability}")

    print("\n--- Morphological Complexity Analysis ---")
    morph_complexity = nlu_module.analyze_morphological_complexity(test_text)
    print(f"Morphological Complexity: {morph_complexity}")

    print("\n--- Syntactic Complexity Analysis ---")
    synt_complexity = nlu_module.analyze_syntactic_complexity(test_text)
    print(f"Syntactic Complexity: {synt_complexity}")

    print("\n--- Pragmatic Appropriateness Analysis ---")
    pragmatic_appropriateness = nlu_module.analyze_pragmatic_appropriateness(
        test_text, test_context
    )
    print(f"Pragmatic Appropriateness: {pragmatic_appropriateness}")

    print("\n--- Phonological Features Analysis ---")
    phonological_features = nlu_module.analyze_phonological_features(test_text)
    print(f"Phonological Features: {phonological_features}")

    print("\n--- Text Quality Analysis ---")
    text_quality = nlu_module.analyze_text_quality(test_text, test_context)
    print(f"Text Quality: {text_quality}")

    # Train Word2Vec model
    print("\n--- Training Word2Vec Model ---")
    try:
        sentences = []
        # Attempt to split by newlines first, as this is a common structure
        lines = test_text.splitlines()
        if len(lines) > 1:
            for line in lines:
                # Further tokenize each line into words
                sentences.append(
                    [word for word in word_tokenize(line) if word.isalnum()]
                )
        else:
            # If no newlines, attempt to split into sentences using nltk
            try:
                nltk_sentences = nltk.sent_tokenize(test_text)
                for sent in nltk_sentences:
                    sentences.append(
                        [word for word in word_tokenize(sent) if word.isalnum()]
                    )
            except Exception as e:
                logger.warning(
                    f"Error using nltk.sent_tokenize, falling back to word tokenization: {e}"
                )
                sentences = [
                    [word for word in word_tokenize(test_text) if word.isalnum()]
                ]

        if not sentences:
            logger.warning("No sentences found for Word2Vec training.")
        else:
            nlu_module.nlp_processor._train_word2vec(sentences)

    except Exception as e:
        logger.error(f"Error preparing text for Word2Vec training: {e}")
    print("Word2Vec model trained.")

    print("\n--- Analysis After Word2Vec Training ---")

    print("\n--- Key Phrase Extraction ---")
    key_phrases = nlu_module.extract_key_phrases(test_text)
    print(f"Key Phrases: {key_phrases}")

    print("\n--- Sentiment Analysis ---")
    sentiment = nlu_module.analyze_sentiment(test_text)
    print(f"Sentiment: {sentiment}")

    print("\n--- Named Entity Extraction ---")
    entities = nlu_module.extract_named_entities(test_text)
    print(f"Named Entities: {entities}")

    print("\n--- Response Clustering ---")
    clusters = nlu_module.cluster_responses(responses, n_clusters=3)
    print(f"Response Clusters: {clusters}")

    print("\n--- Discourse Structure Analysis ---")
    discourse = nlu_module.analyze_discourse_structure(test_text)
    print(f"Discourse Structure: {discourse}")

    print("\n--- Semantic Similarity Analysis ---")
    similarity = nlu_module.analyze_semantic_similarity(test_text, test_context)
    print(f"Semantic Similarity: {similarity}")

    print("\n--- Lexical Diversity Analysis ---")
    diversity = nlu_module.analyze_lexical_diversity(test_text)
    print(f"Lexical Diversity: {diversity}")

    print("\n--- Readability Analysis ---")
    readability = nlu_module.analyze_readability(test_text)
    print(f"Readability: {readability}")

    print("\n--- Morphological Complexity Analysis ---")
    morph_complexity = nlu_module.analyze_morphological_complexity(test_text)
    print(f"Morphological Complexity: {morph_complexity}")

    print("\n--- Syntactic Complexity Analysis ---")
    synt_complexity = nlu_module.analyze_syntactic_complexity(test_text)
    print(f"Syntactic Complexity: {synt_complexity}")

    print("\n--- Pragmatic Appropriateness Analysis ---")
    pragmatic_appropriateness = nlu_module.analyze_pragmatic_appropriateness(
        test_text, test_context
    )
    print(f"Pragmatic Appropriateness: {pragmatic_appropriateness}")

    print("\n--- Phonological Features Analysis ---")
    phonological_features = nlu_module.analyze_phonological_features(test_text)
    print(f"Phonological Features: {phonological_features}")

    print("\n--- Text Quality Analysis ---")
    text_quality = nlu_module.analyze_text_quality(test_text, test_context)
    print(f"Text Quality: {text_quality}")
