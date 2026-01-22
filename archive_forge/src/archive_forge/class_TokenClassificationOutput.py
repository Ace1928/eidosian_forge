from typing import TYPE_CHECKING, List, TypedDict
class TokenClassificationOutput(TypedDict):
    """Dictionary containing the output of a [`~InferenceClient.token_classification`] task.

    Args:
        entity_group (`str`):
            The type for the entity being recognized (model specific).
        score (`float`):
            The score of the label predicted by the model.
        word (`str`):
            The string that was captured.
        start (`int`):
            The offset stringwise where the answer is located. Useful to disambiguate if word occurs multiple times.
        end (`int`):
            The offset stringwise where the answer is located. Useful to disambiguate if word occurs multiple times.
    """
    entity_group: str
    score: float
    word: str
    start: int
    end: int