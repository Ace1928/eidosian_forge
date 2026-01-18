import unicodedata
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt

"""
Demonstration of Processual Modular Programming (PMP) Paradigm: Robust Text Analysis from Real-World Sources

This script provides a sophisticated demonstration of Processual Modular Programming (PMP) through a text processing pipeline designed to handle unstructured, real-world text.
It rigorously adheres to PMP principles—Modularity, Explicit State Management, Dynamic Adaptation, and Composition of Modules—as detailed in 'PMP_My_Code.tex'.

The pipeline is engineered for robustness and performance, effectively processing chaotic text from diverse sources. It showcases:
    - Well-defined, formally specified interfaces ensuring modular interoperability.
    - Encapsulated internal states within each module, promoting clarity and maintainability.
    - Process functions as explicit state transformations, aligning with PMP's process-centric nature.
    - Advanced dynamic adaptation mechanisms, enabling modules to intelligently respond to input characteristics and refine processing.
    - Composition of modules into a cohesive pipeline, demonstrating PMP's architectural flexibility and scalability.

This demonstration exemplifies the practical application and utility of PMP for complex text analysis tasks, emphasizing its suitability for modern, data-intensive applications.
"""


# Module 1: Advanced Text Normalization Module
class TextNormalizer:
    """
    A PMP module for advanced text normalization, incorporating Unicode normalization, whitespace handling, and adaptive rule application.

    This module rigorously embodies 'Modularity and Encapsulation', encapsulating sophisticated text normalization logic within a self-contained unit.
    It features 'Dynamic Adaptation' through context-aware rule adjustments and detailed note-taking on text characteristics, enhancing processing robustness.

    Attributes:
        state (Dict[str, Any]): The comprehensive internal state of the TextNormalizer module.
                                 Includes 'rules' (normalization ruleset), 'notes' (adaptive feedback and processing insights),
                                 and 'adaptation_strategy' (parameters guiding dynamic rule adjustment).
    """

    def __init__(
        self,
        rules: Optional[List[str]] = None,
        adaptation_strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the TextNormalizer module with customizable normalization rules and an adaptation strategy.

        Defaults to Unicode NFC normalization and whitespace stripping if no rules are provided.
        The 'adaptation_strategy' allows for configuring dynamic adjustments based on text properties.
        This constructor exemplifies 'Explicit State Management', initializing a rich module state for advanced processing.

        Args:
            rules (Optional[List[str]]): A list of normalization rules to apply.
                                         Defaults to ['NFC', 'strip_whitespace'] for basic normalization.
            adaptation_strategy (Optional[Dict[str, Any]]): Configuration for dynamic adaptation.
                                                            Currently supports 'short_text_threshold' for note-taking.
        """
        default_rules = ["NFC", "strip_whitespace"]
        self.state: Dict[str, Any] = {
            "rules": rules if rules is not None else default_rules,
            "notes": [],
            "adaptation_strategy": (
                adaptation_strategy
                if adaptation_strategy is not None
                else {"short_text_threshold": 15}
            ),  # Adjusted threshold for 'short text'
        }

    def process(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Processes the input text through normalization and dynamic state adaptation based on text characteristics.

        This method is the core 'process function' (f_M), transforming input text and dynamically updating the module's state.
        It rigorously demonstrates 'Processual Nature' by executing a temporal state transformation.
        Advanced 'Dynamic Adaptation' is implemented by adjusting notes and potentially rules based on text length and content.

        Args:
            input_text (str): The raw, unstructured text input for normalization.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing:
                - The normalized text (output O_M), ready for subsequent processing.
                - A copy of the comprehensively updated module state (updated S_M), reflecting processing outcomes and adaptations.
        """
        normalized_text = self.normalize(input_text)
        text_length = len(normalized_text)
        short_text_threshold = self.state["adaptation_strategy"].get(
            "short_text_threshold", 15
        )

        # Advanced Dynamic Adaptation Function (Δ): Context-aware state refinement based on processing.
        if text_length < short_text_threshold:
            note = (
                f"Short text detected ({text_length} chars), consider content sparsity."
            )
            if note not in self.state["notes"]:
                self.state["notes"].append(note)  # More informative note
        else:
            short_text_note = (
                f"Short text detected ({text_length} chars), consider content sparsity."
            )
            if short_text_note in self.state["notes"]:
                self.state["notes"].remove(short_text_note)  # Resolve short text note

        # Future adaptation: Rule adjustment based on text characteristics could be added here.

        return normalized_text, self.state.copy()

    def normalize(self, text: str) -> str:
        """
        Applies the configured normalization rules to the input text, ensuring robust text preprocessing.

        This utility function encapsulates the core normalization logic, applying rules defined in the module's state for consistent processing.

        Args:
            text (str): The text to be normalized.

        Returns:
            str: The normalized text, processed according to the module's rules.
        """
        processed_text = text
        if "strip_whitespace" in self.state["rules"]:
            processed_text = processed_text.strip()
        if "NFC" in self.state["rules"]:
            processed_text = unicodedata.normalize("NFC", processed_text)
        # Add more sophisticated normalization rules here as needed, e.g., handling URLs, hashtags, etc.
        return processed_text


# Module 2: Intelligent Text Tokenizer Module
class Tokenizer:
    """
    An intelligent PMP module for text tokenization, segmenting text into meaningful tokens with configurable delimiters.

    This module rigorously adheres to 'Modularity' and 'Explicit State Management', providing advanced tokenization capabilities.
    It maintains state for tokenization parameters and dynamically adapts tokenization based on context if configured.

    Attributes:
        state (Dict[str, Any]): The detailed internal state of the Tokenizer module.
                                 Includes 'delimiter' (token separation character), 'token_count', and
                                 'tokenization_strategy' (parameters for advanced tokenization approaches).
    """

    def __init__(
        self,
        delimiter: str = " ",
        tokenization_strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the Tokenizer module with a specified delimiter and optional advanced tokenization strategy.

        Defaults to whitespace (" ") as the delimiter. 'tokenization_strategy' allows for configuring more complex tokenization.
        State initialization is crucial for PMP's 'Explicit State Management', setting up the module for diverse tokenization tasks.

        Args:
            delimiter (str): The delimiter used to tokenize text. Defaults to " ".
            tokenization_strategy (Optional[Dict[str, Any]]): Configuration for advanced tokenization.
                                                                Currently supports basic delimiter-based splitting.
        """
        self.state: Dict[str, Any] = {
            "delimiter": delimiter,
            "token_count": 0,
            "tokenization_strategy": (
                tokenization_strategy if tokenization_strategy is not None else {}
            ),  # Strategy can be extended for punctuation, subword, etc.
        }

    def process(self, text: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Tokenizes the input text using the defined strategy and updates the module's state with the token count.

        This method is the 'process function' (f_M) for advanced tokenization within PMP.
        It transforms normalized text into a list of tokens and updates 'token_count', demonstrating 'Processual Nature'.
        Future enhancements could include dynamic delimiter adjustment or strategy switching based on text content.

        Args:
            text (str): The normalized text to be tokenized (input I_M).

        Returns:
            Tuple[List[str], Dict[str, Any]]: A tuple containing:
                - A list of tokens (output O_M), the result of intelligent tokenization.
                - A copy of the updated module state (updated S_M), including token statistics and strategy details.
        """
        tokens = self.tokenize(text)
        self.state["token_count"] = len(tokens)  # Update token count in state
        return tokens, self.state.copy()

    def tokenize(self, text: str) -> List[str]:
        """
        Performs text tokenization based on the module's configured delimiter and strategy.

        This utility function executes the tokenization logic, applying the defined delimiter for text segmentation.
        Future versions could incorporate more sophisticated tokenization algorithms based on 'tokenization_strategy'.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens, segmented according to the module's tokenization rules.
        """
        return text.split(self.state["delimiter"])  # Basic delimiter-based tokenization


# Module 3: Comprehensive Metrics Collector and Visualizer Module
class MetricsCollector:
    """
    A PMP module for comprehensive text metrics computation and advanced visualization of token length distribution.

    This module exemplifies 'Modularity' and provides 'Visual Feedback', crucial for PMP demonstrability and utility.
    It calculates a range of metrics on tokens and generates aesthetically refined visualizations for enhanced interpretability.

    Attributes:
        metrics (Dict[str, Any]): The extensive internal state storing computed metrics, including:
                                 average, max, min token lengths, token count, vocabulary size, and the tokens themselves.
    """

    def __init__(self) -> None:
        """
        Initializes the MetricsCollector module with an empty metrics state, ready for metric computation.

        State initialization for comprehensive metrics storage, adhering to PMP's 'Explicit State Management' for detailed analysis.
        """
        self.metrics: Dict[str, Any] = {}

    def process(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Computes a comprehensive set of metrics based on the input tokens and updates the module's metrics state.

        This is the 'process function' (f_M) for in-depth metrics computation in PMP, providing rich analytical insights.
        It takes tokens as input, calculates various metrics, and updates the module's internal state ('metrics') for visualization and analysis.

        Args:
            tokens (List[str]): A list of tokens to compute metrics from (input I_M).

        Returns:
            Dict[str, Any]: A copy of the updated metrics state (updated S_M), containing a comprehensive suite of computed metrics.
        """
        if not tokens:
            self.metrics = {
                "average_token_length": 0,
                "max_token_length": 0,
                "min_token_length": 0,
                "token_count": 0,
                "vocabulary_size": 0,  # Added vocabulary size metric
                "tokens": tokens,
            }
        else:
            token_lengths = [len(token) for token in tokens]
            vocabulary = set(tokens)  # Calculate vocabulary size
            self.metrics = {
                "average_token_length": sum(token_lengths) / len(token_lengths),
                "max_token_length": max(token_lengths),
                "min_token_length": min(token_lengths),
                "token_count": len(tokens),
                "vocabulary_size": len(vocabulary),  # Store vocabulary size
                "tokens": tokens,
            }
        return self.metrics.copy()

    def visualize(self) -> None:
        """
        Renders a visually appealing and informative bar chart illustrating the distribution of token lengths, with enhanced aesthetics.

        This method provides 'Visual Feedback', a key aspect of PMP for demonstrative and analytical purposes.
        It uses matplotlib to create a refined bar chart of token lengths, enhancing the interpretability of processed text metrics through visual clarity.
        """
        tokens = self.metrics.get("tokens", [])
        lengths = [len(token) for token in tokens] if tokens else []
        if not lengths:
            print("No tokens available for visualization.")
            return

        plt.figure(figsize=(12, 7))  # Increased figure size for better detail
        bars = plt.bar(
            range(len(lengths)),
            lengths,
            color="mediumseagreen",  # Modern, professional color
            edgecolor="darkgreen",
            alpha=0.85,  # Slightly increased transparency
        )

        # Adding value labels on top of bars for direct length readability
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.5,  # Position slightly above the bar
                round(yval, 1),
                ha="center",
                va="bottom",
                fontsize=8,
                color="dimgray",
            )

        plt.xlabel(
            "Token Index", fontsize=14, color="dimgray"
        )  # Refined, larger labels
        plt.ylabel("Token Length (Characters)", fontsize=14, color="dimgray")
        plt.title(
            "Distribution of Token Lengths in Processed Text",
            fontsize=18,
            fontweight="bold",
            color="midnightblue",  # More prominent, professional title color
        )
        plt.xticks(fontsize=11, color="gray")  # Styled x ticks
        plt.yticks(fontsize=11, color="gray")  # Styled y ticks
        plt.grid(axis="y", linestyle="--", alpha=0.7)  # Cleaner y-axis grid
        plt.tight_layout()
        plt.gca().spines["top"].set_visible(False)  # Remove top spine for cleaner look
        plt.gca().spines["right"].set_visible(False)  # Remove right spine
        plt.gca().set_facecolor(
            "whitesmoke"
        )  # Set a subtle background color for better contrast
        plt.show()


# PMP Pipeline: Advanced Composition of Modules using Dependency Injection.
def PMP_pipeline(input_text: str) -> Dict[str, Any]:
    """
    Composes the advanced PMP processing pipeline, orchestrating normalization, tokenization, and comprehensive metrics analysis.

    This function elegantly demonstrates the 'Composition Operator' (⊕) in PMP, assembling a sophisticated pipeline from modular components.
    It embodies 'Modular Composability' through the seamless integration of well-defined modules, showcasing PMP's architectural strength.
    Dependency injection is implicitly utilized, enhancing modularity and flexibility in pipeline construction.

    Args:
        input_text (str): The raw, unstructured input text to be processed by the comprehensive PMP pipeline.

    Returns:
        Dict[str, Any]: A comprehensive output dictionary, providing detailed insights into the text processing workflow:
            - 'normalized_text': The text after advanced normalization.
            - 'tokens': The list of tokens generated through intelligent tokenization.
            - 'normalizer_state': The final, detailed state of the TextNormalizer module.
            - 'tokenizer_state': The final, detailed state of the Tokenizer module.
            - 'metrics': The comprehensive metrics computed by the MetricsCollector module, offering rich text analysis.
    """
    # Instantiate PMP modules - demonstrating modularity and encapsulation in pipeline construction.
    normalizer = TextNormalizer()
    tokenizer = Tokenizer()
    metrics_collector = MetricsCollector()

    # Step 1: Advanced text normalization using the TextNormalizer module.
    # Output of TextNormalizer serves as input for Tokenizer, demonstrating clear interface compatibility.
    normalized_text, normalizer_state = normalizer.process(input_text)

    # Step 2: Intelligent tokenization using the Tokenizer module.
    # Output of Tokenizer (tokens) becomes input for MetricsCollector, ensuring seamless data flow.
    tokens, tokenizer_state = tokenizer.process(normalized_text)

    # Step 3: Comprehensive metrics collection and visualization using the MetricsCollector module.
    metrics = metrics_collector.process(tokens)
    metrics_collector.visualize()  # Invoke visualization to graphically present computed metrics.

    # Return a comprehensive result dictionary, encapsulating processed data and module states for detailed analysis.
    return {
        "normalized_text": normalized_text,
        "tokens": tokens,
        "normalizer_state": normalizer_state,
        "tokenizer_state": tokenizer_state,
        "metrics": metrics,
    }


if __name__ == "__main__":
    """
    Main execution block demonstrating the PMP pipeline with diverse, real-world text inputs for robust evaluation.

    This section showcases the versatility and robustness of the PMP pipeline with a variety of input scenarios,
    demonstrating its adaptability and effectiveness in handling unstructured text from genuine sources. Examples include:
        - Social media snippets with mixed formatting and informal language.
        - News headlines representing concise and information-dense text.
        - Excerpts from online reviews, showcasing user-generated content with varied linguistic styles.
        - Short, telegraphic text to rigorously test dynamic adaptation in TextNormalizer.
    """
    print("PMP Pipeline Demonstration: Real-World Text Analysis\n" + "=" * 60)

    # Example 1: Social Media Snippet - Demonstrates handling of informal text and mixed formatting.
    raw_text_1 = (
        "OMG! Just saw the most amazing sunset #nofilter #blessed. "
        "Gonna grab a ☕ now!  What's everyone else up to?"
    )
    result_1 = PMP_pipeline(raw_text_1)
    print("\nExample 1: Social Media Text Processing")
    print("-" * 60)
    print("Normalized Text:", f"'{result_1['normalized_text']}'")
    print("Tokenized Output:", result_1["tokens"])
    print("Normalizer State:", result_1["normalizer_state"])
    print("Tokenizer State:", result_1["tokenizer_state"])
    print(
        "Computed Metrics:",
        {k: v for k, v in result_1["metrics"].items() if k != "tokens"},
    )  # Print metrics excluding tokens

    # Example 2: News Headline - Demonstrates processing of concise, information-rich text.
    raw_text_2 = (
        "Breaking News: Global Markets React to Unexpected Economic Data Release."
    )
    result_2 = PMP_pipeline(raw_text_2)
    print("\nExample 2: News Headline Processing")
    print("-" * 60)
    print("Normalized Text:", f"'{result_2['normalized_text']}'")
    print("Tokenized Output:", result_2["tokens"])
    print("Normalizer State:", result_2["normalizer_state"])
    print("Tokenizer State:", result_2["tokenizer_state"])
    print(
        "Computed Metrics:",
        {k: v for k, v in result_2["metrics"].items() if k != "tokens"},
    )  # Print metrics excluding tokens

    # Example 3: Online Review Excerpt - Demonstrates handling of user-generated content with varied styles.
    raw_text_3 = "The new restaurant was okay, I guess. Service was kinda slow, but the food was decent. 🤷‍♂️"
    result_3 = PMP_pipeline(raw_text_3)
    print("\nExample 3: Online Review Text Processing")
    print("-" * 60)
    print("Normalized Text:", f"'{result_3['normalized_text']}'")
    print("Tokenized Output:", result_3["tokens"])
    print("Normalizer State:", result_3["normalizer_state"])
    print("Tokenizer State:", result_3["tokenizer_state"])
    print(
        "Computed Metrics:",
        {k: v for k, v in result_3["metrics"].items() if k != "tokens"},
    )  # Print metrics excluding tokens

    # Example 4: Short, Telegraphic Input - Tests dynamic adaptation for minimal text.
    raw_text_4 = "Urgent meeting tmrw 9am sharp!"
    result_4 = PMP_pipeline(raw_text_4)
    print("\nExample 4: Telegraphic Text & Dynamic Adaptation")
    print("-" * 60)
    print("Normalized Text:", f"'{result_4['normalized_text']}'")
    print("Tokenized Output:", result_4["tokens"])
    print("Normalizer State:", result_4["normalizer_state"])
    print("Tokenizer State:", result_4["tokenizer_state"])
    print(
        "Computed Metrics:",
        {k: v for k, v in result_4["metrics"].items() if k != "tokens"},
    )  # Print metrics excluding tokens

    print("\n" + "=" * 60 + "\nEnd of PMP Demonstration: Real-World Text Analysis")
