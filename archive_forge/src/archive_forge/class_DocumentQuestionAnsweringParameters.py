from dataclasses import dataclass
from typing import Any, List, Optional, Union
from .base import BaseInferenceType
@dataclass
class DocumentQuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Document Question Answering
    """
    doc_stride: Optional[int] = None
    'If the words in the document are too long to fit with the question for the model, it will\n    be split in several chunks with some overlap. This argument controls the size of that\n    overlap.\n    '
    handle_impossible_answer: Optional[bool] = None
    'Whether to accept impossible as an answer'
    lang: Optional[str] = None
    'Language to use while running OCR. Defaults to english.'
    max_answer_len: Optional[int] = None
    'The maximum length of predicted answers (e.g., only answers with a shorter length are\n    considered).\n    '
    max_question_len: Optional[int] = None
    'The maximum length of the question after tokenization. It will be truncated if needed.'
    max_seq_len: Optional[int] = None
    'The maximum length of the total sentence (context + question) in tokens of each chunk\n    passed to the model. The context will be split in several chunks (using doc_stride as\n    overlap) if needed.\n    '
    top_k: Optional[int] = None
    'The number of answers to return (will be chosen by order of likelihood). Can return less\n    than top_k answers if there are not enough options available within the context.\n    '
    word_boxes: Optional[List[Union[List[float], str]]] = None
    'A list of words and bounding boxes (normalized 0->1000). If provided, the inference will\n    skip the OCR step and use the provided bounding boxes instead.\n    '