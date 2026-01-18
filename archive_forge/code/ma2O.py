from typing import Dict, List
import spacy
from langdetect import detect


class NLPNLUProcessor:
    """
    Provides advanced NLP and NLU functionalities to process and understand text data,
    integrating natural language understanding capabilities. This class utilizes the spaCy library
    for NLP tasks and langdetect for language detection, supporting multiple languages with specific
    models loaded dynamically based on language detection.

    Attributes:
        nlp_models (Dict[str, spacy.lang]): Dictionary of loaded spaCy language models.
    """

    def __init__(self):
        """
        Initializes the NLP/NLU processing capabilities. Loads spaCy language models that are necessary
        for processing different languages and tasks, initially for English and Spanish.
        """
        self.nlp_models = {
            "en": spacy.load("en_core_web_md"),  # English medium model
            "es": spacy.load("es_core_news_md"),  # Spanish medium model
        }

    def process_text(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Processes text to detect its language, and then extract and return advanced linguistic features including
        tokens, entities, and sentence structures.

        :param text: str - Text to be processed.
        :return: Dict[str, List[Dict[str, str]]] - A dictionary containing tokens, entities, and sentences.
        """
        language = self.language_detection(text)
        nlp = self.nlp_models.get(
            language, self.nlp_models["en"]
        )  # Default to English if model not found
        return self._enhance_text_data(text, nlp)

    def _enhance_text_data(self, text: str, nlp) -> Dict[str, List[Dict[str, str]]]:
        """
        Extracts linguistic features using the specified spaCy model.

        :param text: str - The text to be processed.
        :param nlp: spacy.Language - spaCy model for text processing.
        :return: Dict[str, List[Dict[str, str]]] - Processed features including tokens, entities, sentences.
        """
        doc = nlp(text)
        tokens = [
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
            }
            for token in doc
        ]
        entities = [{"text": entity.text, "type": entity.label_} for entity in doc.ents]
        sentences = [sent.text for sent in doc.sents]
        return {"tokens": tokens, "entities": entities, "sentences": sentences}

    def language_detection(self, text: str) -> str:
        """
        Determines the language of the given text using the langdetect library.

        :param text: str - Text whose language is to be detected.
        :return: str - ISO 639-1 language code (e.g., 'en' for English, 'es' for Spanish).
        """
        return detect(text)

    def named_entity_recognition(self, text: str) -> List[Dict[str, str]]:
        """
        Identifies and classifies named entities in the text using the detected language model.

        :param text: str - Text to be analyzed for named entities.
        :return: List[Dict[str, str]] - A list of entities with text and type.
        """
        language = self.language_detection(text)
        nlp = self.nlp_models.get(
            language, self.nlp_models["en"]
        )  # Default to English if model not found
        doc = nlp(text)
        return [{"text": entity.text, "type": entity.label_} for entity in doc.ents]
