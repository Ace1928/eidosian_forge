from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def text_state_params(self, value: Union[bytes, str]='') -> TextStateParams:
    """
        Create a TextStateParams instance to display a text string. Type[bytes] values
        will be decoded implicitly.

        Args:
            value (str | bytes): text to associate with the captured state.

        Raises:
            PdfReadError: if font not set (no Tf operator in incoming pdf content stream)

        Returns:
            TextStateParams: current text state parameters
        """
    if not isinstance(self.font, Font):
        raise PdfReadError('font not set: is PDF missing a Tf operator?')
    if isinstance(value, bytes):
        try:
            if isinstance(self.font.encoding, str):
                txt = value.decode(self.font.encoding, 'surrogatepass')
            else:
                txt = ''.join((self.font.encoding[x] if x in self.font.encoding else bytes((x,)).decode() for x in value))
        except (UnicodeEncodeError, UnicodeDecodeError):
            txt = value.decode('utf-8', 'replace')
        txt = ''.join((self.font.char_map[x] if x in self.font.char_map else x for x in txt))
    else:
        txt = value
    return TextStateParams(txt, self.font, self.font_size, self.Tc, self.Tw, self.Tz, self.TL, self.Ts, self.effective_transform)