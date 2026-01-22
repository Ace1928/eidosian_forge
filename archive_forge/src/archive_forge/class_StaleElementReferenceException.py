from typing import Optional
from typing import Sequence
class StaleElementReferenceException(WebDriverException):
    """Thrown when a reference to an element is now "stale".

    Stale means the element no longer appears on the DOM of the page.


    Possible causes of StaleElementReferenceException include, but not limited to:
        * You are no longer on the same page, or the page may have refreshed since the element
          was located.
        * The element may have been removed and re-added to the screen, since it was located.
          Such as an element being relocated.
          This can happen typically with a javascript framework when values are updated and the
          node is rebuilt.
        * Element may have been inside an iframe or another context which was refreshed.
    """

    def __init__(self, msg: Optional[str]=None, screen: Optional[str]=None, stacktrace: Optional[Sequence[str]]=None) -> None:
        with_support = f'{msg}; {SUPPORT_MSG} {ERROR_URL}#stale-element-reference-exception'
        super().__init__(with_support, screen, stacktrace)