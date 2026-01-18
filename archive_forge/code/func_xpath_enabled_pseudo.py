import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_enabled_pseudo(self, xpath: XPathExpr) -> XPathExpr:
    return xpath.add_condition("\n        (\n            @href and (\n                name(.) = 'a' or\n                name(.) = 'link' or\n                name(.) = 'area'\n            )\n        ) or (\n            (\n                name(.) = 'command' or\n                name(.) = 'fieldset' or\n                name(.) = 'optgroup'\n            )\n            and not(@disabled)\n        ) or (\n            (\n                (name(.) = 'input' and @type != 'hidden') or\n                name(.) = 'button' or\n                name(.) = 'select' or\n                name(.) = 'textarea' or\n                name(.) = 'keygen'\n            )\n            and not (@disabled or ancestor::fieldset[@disabled])\n        ) or (\n            name(.) = 'option' and not(\n                @disabled or ancestor::optgroup[@disabled]\n            )\n        )\n        ")