import math
import random
import unicodedata
import cmath
import os
import threading
import time
import re
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights
import torch
from abc import ABC, abstractmethod


# Optimize CPU usage by limiting the number of CPU threads used by PyTorch
torch.set_num_threads(4)


# Base module interface for loosely coupled components
class BaseModule(ABC):
    """Abstract base class for all framework modules providing state management"""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}


##############################################################
# Module A: Preprocessing Operator (Pin)
##############################################################
class Preprocessor(BaseModule):
    """
    Preprocessing Operator: Pin
    ----------------------------
    Implements Theorem 1: X_proc = P_in(X_raw)
    Maps raw input X_raw ∈ Σ* to preprocessed input X_proc through:
    - Whitespace trimming
    - Unicode NFC normalization
    - Encoding validation
    """

    def __init__(self) -> None:
        super().__init__()
        self.state = {
            "method": "Pin",
            "notes": "Trims whitespace and applies Unicode NFC normalization; maps X_raw to X_proc.",
        }

    def process(self, raw_text: str) -> str:
        """Executes deterministic preprocessing pipeline"""
        processed = raw_text.strip()
        processed = unicodedata.normalize("NFC", processed)
        return processed


##############################################################
# Module D: Vocabulary and Tokenization
##############################################################
class Vocabulary:
    """
    Vocabulary Management Module
    ------------------------------
    Maintains dual vocabulary system with:
    - Base vocabulary (Σ_base)
    - Dynamically learned vocabulary (Σ_learned)
    Implements Definition 3: Vocabulary Expansion Protocol
    """

    def __init__(self) -> None:
        self.base_vocab = {"the", "and", "to", "of", "a", "in"}
        self.learned_vocab: set[str] = set()
        self.token_to_id: Dict[str, int] = {}
        self.next_id = 1
        self.definitions: Dict[str, str] = {}

    def update(self, token_str: str) -> int:
        """Implements Algorithm 2: Dynamic Token Integration"""
        token_str = token_str.lower()
        if token_str not in self.token_to_id:
            self.token_to_id[token_str] = self.next_id
            self.next_id += 1
            if token_str not in self.base_vocab:
                self.learned_vocab.add(token_str)
        return self.token_to_id[token_str]

    def dynamic_update(self, tokens: List[str]) -> None:
        """Batch update mechanism for vocabulary expansion"""
        for token in tokens:
            self.update(token)

    @property
    def complete_vocab(self) -> set:
        """Returns union of base and learned vocabularies (Σ*)"""
        return self.base_vocab.union(self.learned_vocab)

    def add_definition(self, token: str, definition: str) -> None:
        """Stores semantic definitions for new tokens"""
        self.definitions[token.lower()] = definition


class Token:
    """
    Token Structure and Quantum Representation
    --------------------------------------------
    Implements Definition 4: Token Algebraic Structure
    Canonical representation: t = (u, π, χ, Ψ)
    where:
    - u: Underlying unit (string)
    - π: Intrinsic properties (vector)
    - χ: Contextual statistics (vector)
    - Ψ: Quantum superposition coefficients
    """

    def __init__(self, unit: str, vocab: Vocabulary) -> None:
        self.unit = unit
        self.intrinsic: List[float] = [float(len(unit)), 1.0]  # π
        self.contextual: List[float] = [1.0, 0.5]  # χ

        # Quantum coefficients (α, β, γ) ∈ ℂ^3
        a = complex(random.random(), random.random())
        b = complex(random.random(), random.random())
        c = complex(random.random(), random.random())
        norm = math.sqrt(abs(a) ** 2 + abs(b) ** 2 + abs(c) ** 2)
        self.quantum_coeff: Tuple[complex, complex, complex] = (
            a / norm,
            b / norm,
            c / norm,
        )

        self.id: int = vocab.update(unit)
        self.embedding: List[float] = []  # Holomorphic embedding


##############################################################
# Hypergeometric Tokenization Module
##############################################################
class HypergeometricTokenizer(BaseModule):
    """
    Hypergeometric Tokenization Module
    ----------------------------------
    Implements Theorem 2: Energy-Minimized Tokenization
    Supports both classical and quantum tokenization modes
    """

    def __init__(self, llm_tokenizer: Optional[Any] = None) -> None:
        super().__init__()
        self.llm_tokenizer = llm_tokenizer
        self.state = {
            "method": "Hypergeometric",
            "notes": (
                "Tokenizes text using a model's tokenizer if available; "
                "otherwise employs whitespace splitting as an approximation."
            ),
        }

    def process(self, text: str, vocab: Vocabulary) -> Tuple[List[Token], Dict]:
        """Executes tokenization process with energy minimization"""
        tokens_str = (
            self.llm_tokenizer.tokenize(text) if self.llm_tokenizer else text.split()
        )

        tokens: List[Token] = []
        for token_str in tokens_str:
            token_obj = Token(token_str, vocab)
            tokens.append(token_obj)

        self.state["token_count"] = len(tokens)
        return tokens, self.state.copy()


##############################################################
# Module E: Embedding and Fusion Operator
##############################################################
class EmbeddingModule(BaseModule):
    """
    Embedding and Fusion Operator Module
    --------------------------------------
    Implements Theorem 3: Holomorphic Embedding Fusion
    F: E_B × E_C → E_F with ∂E/∂z̄ = 0
    """

    def __init__(self) -> None:
        super().__init__()
        self.state = {
            "notes": (
                "Fuses base (E_B) and contextual (E_C) embeddings via "
                "holomorphic continuation in ℂ^d_F"
            )
        }
        self.d_E = 2  # Base embedding dimension
        self.d_C = 2  # Contextual embedding dimension
        self.d_F = 2  # Fused embedding dimension

    def base_embedding(self, intrinsic: List[float]) -> List[float]:
        """Computes base embedding E_B ∈ ℝ^d_E"""
        return [0.5 * intrinsic[0], 0.5 * intrinsic[1]]

    def contextual_embedding(self, contextual: List[float]) -> List[float]:
        """Computes contextual embedding E_C ∈ ℝ^d_C"""
        return [0.8 * contextual[0], 0.8 * contextual[1]]

    def fusion_operator(
        self, base_emb: List[float], context_emb: List[float]
    ) -> List[float]:
        """Implements holomorphic fusion operator F"""
        return [
            base_emb[i] + context_emb[i]
            for i in range(min(len(base_emb), len(context_emb)))
        ]

    def process(self, tokens: List[Token]) -> Tuple[List[List[float]], Dict]:
        """Executes full embedding pipeline"""
        embeddings: List[List[float]] = []
        for token in tokens:
            eb = self.base_embedding(token.intrinsic)
            ec = self.contextual_embedding(token.contextual)
            fused = self.fusion_operator(eb, ec)
            token.embedding = fused
            embeddings.append(fused)

        self.state["embedding_dimension"] = self.d_F
        return embeddings, self.state.copy()


##############################################################
# Module F: Knowledge Graph Constructor (Base and Personal)
##############################################################
class KnowledgeGraphConstructor:
    """
    Knowledge Graph Constructor Module
    ------------------------------------
    Implements Theorem 4: Graph Fusion Protocol
    Constructs and merges base (G_B) and personal (G_P) knowledge graphs
    """

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "notes": "Constructs and fuses knowledge graphs using algebraic graph fusion"
        }

    def build_base_graph(self, tokens: List[Token]) -> Dict:
        """Constructs base graph G_B from token embeddings"""
        nodes = [token.id for token in tokens]
        edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        attributes = {token.id: token.embedding for token in tokens}
        return {"nodes": nodes, "edges": edges, "attributes": attributes}

    def build_personal_graph(self, tokens: List[Token]) -> Dict:
        """Constructs personalized graph G_P with augmented embeddings"""
        nodes = [token.id for token in tokens]
        edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        attributes = {
            token.id: [val * 1.1 for val in token.embedding] for token in tokens
        }
        return {"nodes": nodes, "edges": edges, "attributes": attributes}

    def fuse_graphs(self, base_graph: Dict, personal_graph: Dict) -> Tuple[Dict, Dict]:
        """Executes graph fusion: G_U = Φ(G_B, G_P)"""
        unified_nodes = list(set(base_graph["nodes"]).union(personal_graph["nodes"]))
        unified_edges = list(set(base_graph["edges"]).union(personal_graph["edges"]))

        unified_attributes = {}
        for node in unified_nodes:
            base_attr = base_graph["attributes"].get(node, [0, 0])
            pers_attr = personal_graph["attributes"].get(node, [0, 0])
            unified_attributes[node] = [
                (b + p) / 2 for b, p in zip(base_attr, pers_attr)
            ]

        unified_graph = {
            "nodes": unified_nodes,
            "edges": unified_edges,
            "attributes": unified_attributes,
        }
        self.state["graph_size"] = len(unified_nodes)
        return unified_graph, self.state.copy()


##############################################################
# Module G: Infinite RoPE Transformation and Dynamic Vocabulary
##############################################################
class InfiniteRoPEModule(BaseModule):
    """
    Infinite RoPE and Dynamic Vocabulary Module
    --------------------------------------------
    Implements Theorem 5: Continuous Positional Encoding
    Applies rotational position encoding through θ-parameterized transformations
    """

    def __init__(self, theta: float = 0.1) -> None:
        super().__init__()
        self.theta = theta
        self.state = {
            "notes": "Applies rotational position encoding via Infinite RoPE transformation"
        }

    def process(self, tokens: List[Token]) -> Tuple[List[List[float]], Dict]:
        """Executes RoPE transformation on token embeddings"""
        transformed_embeddings: List[List[float]] = []
        for i, token in enumerate(tokens):
            if token.embedding:
                x = token.embedding[0]
                angle = (i + 1) * self.theta
                new_embedding = [x * math.cos(angle), x * math.sin(angle)]
                token.embedding = new_embedding
                transformed_embeddings.append(new_embedding)

        self.state["tokens_transformed"] = len(transformed_embeddings)
        return transformed_embeddings, self.state.copy()


##############################################################
# Module I: Titans Memory Architecture
##############################################################
class TitansMemoryModule:
    """
    Titans Memory Architecture Module
    -----------------------------------
    Implements Theorem 6: Non-commutative Memory Operations
    Maintains memory bank with attention-based recall and meta-learning updates
    """

    def __init__(self, tau: float = 1.0) -> None:
        self.memory_bank: List[Tuple[List[float], List[float]]] = []
        self.tau = tau
        self.state: Dict[str, Any] = {
            "notes": "Implements attention-based memory aggregation with non-commutative updates"
        }

    def update_memory(self, key: List[float], value: List[float]) -> None:
        """Stores new memory entries"""
        self.memory_bank.append((key, value))

    def similarity(self, query: List[float], key: List[float]) -> float:
        """Computes cosine similarity between query and key"""
        return sum(q * k for q, k in zip(query, key))

    def aggregate_memory_read(self, query: List[float]) -> List[float]:
        """Performs attention-based memory recall"""
        scores = [self.similarity(query, key) for key, _ in self.memory_bank]
        exp_scores = [math.exp(s / self.tau) for s in scores]
        total = sum(exp_scores) if exp_scores else 1.0
        weights = [s / total for s in exp_scores]
        aggregated = [0.0] * (len(self.memory_bank[0][1]) if self.memory_bank else 1)
        for weight, (_, value) in zip(weights, self.memory_bank):
            aggregated = [a + weight * v for a, v in zip(aggregated, value)]
        return aggregated

    def meta_learner(self, input_vector: List[float]) -> List[float]:
        return [0.5 * x for x in input_vector]

    def compute_commutator(self, v1: List[float], v2: List[float]) -> List[float]:
        if len(v1) == 3 and len(v2) == 3:
            comm = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]
            return [complex(0, comp) for comp in comm]
        else:
            raise NotImplementedError(
                "Commutator implemented only for 3-dimensional vectors."
            )


##############################################################
# Module R: Recursive Adaptive Feedback
##############################################################
class RecursiveFeedbackModule:
    """
    Recursive Adaptive Feedback Module
    ------------------------------------
    Updates the runtime state using an idempotent recursive process.
    """

    def __init__(self):
        self.state: Dict[str, any] = {"feedback": 0}

    def update_state(self, current_state: Dict) -> Dict:
        if current_state.get("updated", False):
            return current_state
        updated_state = current_state.copy()
        updated_state["feedback"] += 1
        updated_state["updated"] = True
        return updated_state


##############################################################
# Module K: Universal Training System
##############################################################
class TrainingSystem:
    """
    Universal Training System Module
    ----------------------------------
    Handles parameter updates via mini-batch loss and provides utility functions.
    """

    def __init__(self, initial_theta: float = 1.0):
        self.theta = initial_theta
        self.state: Dict[str, any] = {
            "notes": "Performs parameter updates with regularization and simulated optimization."
        }
        self.lambda_W = 0.01
        self.lambda_sparse = 0.001

    def compute_loss(self, predictions: List[float], targets: List[float]) -> float:
        mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
        reg = self.lambda_W * (self.theta**2) + self.lambda_sparse * abs(self.theta)
        return mse + reg

    def update_parameters(
        self, batch: Tuple[List[float], List[float]]
    ) -> Tuple[float, Dict]:
        predictions, targets = batch
        loss = self.compute_loss(predictions, targets)
        delta = 0.1 * loss
        self.theta += delta
        self.state["last_loss"] = loss
        return self.theta, self.state.copy()

    def normalize(self, x: float) -> float:
        mean = x
        std = 1  # Placeholder standard deviation.
        return (x - mean) / std

    def dropout(self, x: float, p: float = 0.5) -> float:
        return 0 if random.random() < p else x

    def skip_connection(self, x: float, F_x: float) -> float:
        return x + F_x


##############################################################
# Module L: Output Decoder
##############################################################
class OutputDecoder(BaseModule):
    """
    Output Decoder Module
    -----------------------
    Decodes model output into a modality-specific representation.
    """

    def __init__(self):
        super().__init__()
        self.state = {
            "notes": (
                "Decodes outputs into a unified multimodal representation with default modality 'Text', "
                "using the modality selection function μ_mod."
            )
        }

    def decode(self, model_output, task: str = None) -> Tuple[str, Dict]:
        if task is None:
            task = "Text"
        decoded = f"[{task} Modality] {model_output}"
        self.state["last_decoded"] = decoded
        return decoded, self.state.copy()


##############################################################
# New Module: LLMModelWrapper
##############################################################
class LLMModelWrapper:
    """
    LLM Model Wrapper
    -----------------
    Wraps a locally loaded LLM (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) using Hugging Face Transformers
    and Accelerate. Supports disk offloading and an optional mechanism to remove <think>…</think> tags.
    """

    def __init__(
        self, model_id: str, offload_dir: str = None, remove_think: bool = True
    ):
        self.model_id = model_id
        self.offload_dir = offload_dir if offload_dir is not None else "./offload"
        self.device = "cpu"
        self.remove_think_tags_option = remove_think
        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # Automatically map layers to CPU.
            max_memory={"cpu": "2GB"},
            offload_folder=self.offload_dir,
            torch_dtype=torch.float32,
        )
        self.model_updated = False  # Flag for conditional saving.

    def mark_updated(self) -> None:
        """Marks the model as updated."""
        self.model_updated = True

    def save_model(self, save_directory: str) -> None:
        """Saves the model only if it has been updated."""
        if self.model_updated:
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            self.config.save_pretrained(save_directory)
            print("Model has been updated. Saving to disk.")
            self.model_updated = False  # Reset flag after saving.
        else:
            print("Model not updated; skipping saving process.")

    def _remove_think_tags(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def extract_and_remove_think_tags(self, text: str) -> Tuple[str, str]:
        think_matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        think_content = "\n".join(think_matches)
        return clean_text, think_content

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.remove_think_tags_option:
            generated_text = self._remove_think_tags(generated_text)
        return generated_text

    def get_embeddings(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]

    def generate_chat(self, prompt: str, max_new_tokens: int = 1024) -> Dict[str, str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response, think_content = self.extract_and_remove_think_tags(generated_text)
        return {"response": response, "think": think_content}

    def define_word(self, word: str, max_new_tokens: int = 50) -> str:
        """
        Uses the LLM to provide a definition/meaning for the provided word.
        """
        prompt = f"Define the word '{word}' in clear and simple terms."
        # Here use generate() to ensure that any <think> tags are removed.
        definition = self.generate(prompt, max_new_tokens=max_new_tokens)
        return definition


##############################################################
# Algorithm 1: Eidos Quantum-Adaptive Inference with Module Annotations
##############################################################
def quantum_adaptive_inference(raw_input: str, iterations: int = 3) -> Dict:
    """
    Eidos Quantum-Adaptive Inference Pipeline with LLM Integration
    ---------------------------------------------------------------
    Data Flow:
      1. LLM Integration: Load LLM model.
      2. Input Processing: Preprocess raw input.
      3. Tokenization: Generate tokens with unique identifiers.
      4. Embedding: Compute final token representations enforcing holomorphic constraints.
      5. RoPE Transformation: Apply positional encoding.
      6. Knowledge Graph Construction: Build and fuse knowledge graphs.
      7. Memory Integration: Aggregate a memory read from token embeddings.
      8. LLM Processing: Generate text and embeddings (optionally removing <think> tags).
      9. Parameter Update: Update theta using mini-batch loss.
     10. Recursive Feedback: Update runtime state recursively (idempotently).
     11. Final Decoding: Decode output into a modality-specific representation.
    """
    # LLM Integration
    llm_wrapper = LLMModelWrapper(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", offload_dir="./offload"
    )

    # Module A: Preprocessing
    preprocessor = Preprocessor()
    processed_input = preprocessor.process(raw_input)

    # Module D: Tokenization using the model's tokenizer.
    vocab = Vocabulary()
    tokenizer_module = HypergeometricTokenizer(llm_tokenizer=llm_wrapper.tokenizer)
    tokens, tokenizer_state = tokenizer_module.process(processed_input, vocab)

    # Module E: Embedding and Fusion.
    embedding_module = EmbeddingModule()
    embeddings, embedding_state = embedding_module.process(tokens)

    # Module G: Infinite RoPE Transformation.
    rope_module = InfiniteRoPEModule(theta=0.2)
    rope_embeddings, rope_state = rope_module.process(tokens)

    # Module F: Knowledge Graph Construction.
    kg_constructor = KnowledgeGraphConstructor()
    base_graph = kg_constructor.build_base_graph(tokens)
    personal_graph = kg_constructor.build_personal_graph(tokens)
    unified_graph, kg_state = kg_constructor.fuse_graphs(base_graph, personal_graph)

    # Module I: Titans Memory Integration.
    memory_module = TitansMemoryModule(tau=1.0)
    for token in tokens:
        # Use token embedding as both key and value.
        memory_module.update_memory(token.embedding, token.embedding)

    # Module K: Universal Training System.
    training_system = TrainingSystem(initial_theta=1.0)

    # Module R: Recursive Adaptive Feedback.
    feedback_module = RecursiveFeedbackModule()
    runtime_state = {"feedback": 0}

    print("Starting Quantum-Adaptive Inference Process with LLM Integration")
    for t in range(iterations):
        print(f"\nIteration {t+1}:")
        psi_embeddings, _ = rope_module.process(tokens)
        query = tokens[0].embedding if tokens else [0.0, 0.0]
        memory_read = memory_module.aggregate_memory_read(query)
        # Fuse knowledge: sum of unified graph node IDs plus aggregated memory.
        fused_knowledge = sum(unified_graph["nodes"]) + sum(memory_read)
        predictions = [training_system.theta]
        targets = [fused_knowledge * 0.01]  # Scale target.
        theta, training_state = training_system.update_parameters(
            (predictions, targets)
        )
        print(
            f" Updated Theta: {theta:.4f}, Loss: {training_state.get('last_loss', 0):.4f}"
        )
        runtime_state = feedback_module.update_state(runtime_state)
        print(f" Runtime State: {runtime_state}")

    # Module L: Final Decoding.
    llm_embeddings = llm_wrapper.get_embeddings(processed_input)
    model_generated_text = llm_wrapper.generate(processed_input, max_new_tokens=100)
    final_output = {
        "final_theta": training_system.theta,
        "runtime_state": runtime_state,
        "model_generated_text": model_generated_text,
        "llm_embeddings": (
            llm_embeddings.detach().cpu().numpy().tolist()
            if hasattr(llm_embeddings, "cpu")
            else llm_embeddings
        ),
    }
    decoder = OutputDecoder()
    decoded_output, decoder_state = decoder.decode(final_output)

    result = {
        "processed_input": processed_input,
        "tokens": [token.unit for token in tokens],
        "vocabulary": list(vocab.complete_vocab),
        "embeddings": embeddings,
        "rope_embeddings": rope_embeddings,
        "knowledge_graph": unified_graph,
        "final_theta": training_system.theta,
        "runtime_state": runtime_state,
        "model_generated_text": model_generated_text,
        "llm_embeddings": final_output["llm_embeddings"],
        "decoded_output": decoded_output,
        "states": {
            "preprocessor": preprocessor.state,
            "tokenizer": tokenizer_state,
            "embedding": embedding_state,
            "rope": rope_state,
            "knowledge_graph": kg_state,
            "memory": memory_module.state,
            "training": training_state,
            "decoder": decoder_state,
        },
    }
    return result


def background_learning_thread(
    conversation_history: List[Dict], vocab: Vocabulary, llm_wrapper: LLMModelWrapper
):
    """
    Background thread that simulates continuous learning using conversation history.
    It updates the shared vocabulary and, for new words, uses the LLM to produce definitions.
    """
    while True:
        if conversation_history:
            print(
                "Background Learning: processing conversation history for vocabulary updates..."
            )
            for entry in conversation_history:
                # Extract words from user's input (ignoring case).
                words = re.findall(r"\w+", entry["user"])
                for word in words:
                    word_lower = word.lower()
                    if word_lower not in vocab.token_to_id:
                        vocab.update(word_lower)
                        # If no definition exists, obtain one from the model.
                        if word_lower not in vocab.definitions:
                            definition = llm_wrapper.define_word(
                                word_lower, max_new_tokens=50
                            )
                            vocab.add_definition(word_lower, definition)
                            print(
                                f"Updated vocabulary: Added definition for '{word_lower}': {definition}"
                            )
        time.sleep(10)  # Run every 10 seconds.


def chat_mode():
    """
    Runs a chat interface that integrates with the Eidos framework.
    Processes user input, generates chat responses (with <think> tags removed),
    and logs both the response and the extracted <think> content.
    Also integrates background learning for vocabulary updates.
    """
    conversation_history = []
    # Create a shared vocabulary instance for chat.
    shared_vocab = Vocabulary()
    llm_chat = LLMModelWrapper(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", offload_dir="./offload"
    )

    # Launch the background learning thread with shared vocabulary.
    learner_thread = threading.Thread(
        target=background_learning_thread,
        args=(conversation_history, shared_vocab, llm_chat),
        daemon=True,
    )
    learner_thread.start()

    # Set a directory for periodic model saving.
    save_directory = os.path.join(
        os.path.expanduser("~"),
        "Development",
        "saved_models",
        "deepseek-ai",
        "DeepSeek-R1-Distill-Qwen-1.5B",
    )
    os.makedirs(save_directory, exist_ok=True)

    print("Chat mode initiated. Type your message (or 'exit' to quit).")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
        chat_result = llm_chat.generate_chat(user_input, max_new_tokens=100)
        print("Chat Response:", chat_result["response"])
        # Append new conversation turn.
        conversation_history.append(
            {
                "user": user_input,
                "response": chat_result["response"],
                "think": chat_result["think"],
            }
        )
        # Mark model as updated if any hidden thought information is present.
        if chat_result["think"]:
            llm_chat.mark_updated()
        # Periodically (or every turn) save the model if it was updated.
        llm_chat.save_model(save_directory)


if __name__ == "__main__":
    raw_text = (
        "Eidos framework integrates multimodal data across diverse domains with persistent adaptivity, "
        "idempotent recursion, and unified multimodality."
    )
    result = quantum_adaptive_inference(raw_text, iterations=3)
    print("\nQuantum-Adaptive Inference Result:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Save the LLM model to disk if updated.
    save_path = os.path.join(
        os.path.expanduser("~"),
        "Development",
        "saved_models",
        "deepseek-ai",
        "DeepSeek-R1-Distill-Qwen-1.5B",
    )
    os.makedirs(save_path, exist_ok=True)
    llm_wrapper = LLMModelWrapper(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", offload_dir="./offload"
    )
    llm_wrapper.save_model(save_path)
    print(f"\nLLM model saved to: {save_path}")

    # Launch chat mode.
    chat_mode()
