from typing import TYPE_CHECKING, List, TypedDict
class FillMaskOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.fill_mask`] task.

    Args:
        score (`float`):
            The probability of the token.
        token (`int`):
            The id of the token.
        token_str (`str`):
            The string representation of the token.
        sequence (`str`):
            The actual sequence of tokens that ran against the model (may contain special tokens).
    """
    score: float
    token: int
    token_str: str
    sequence: str