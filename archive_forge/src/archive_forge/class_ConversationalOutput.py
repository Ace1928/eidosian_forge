from typing import TYPE_CHECKING, List, TypedDict
class ConversationalOutput(TypedDict):
    """Dictionary containing the output of a  [`~InferenceClient.conversational`] task.

    Args:
        generated_text (`str`):
            The last response from the model.
        conversation (`ConversationalOutputConversation`):
            The past conversation.
        warnings (`List[str]`):
            A list of warnings associated with the process.
    """
    conversation: ConversationalOutputConversation
    generated_text: str
    warnings: List[str]