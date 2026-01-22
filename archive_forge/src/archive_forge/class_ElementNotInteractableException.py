from typing import Optional
from typing import Sequence
class ElementNotInteractableException(InvalidElementStateException):
    """Thrown when an element is present in the DOM but interactions with that
    element will hit another element due to paint order."""