from typing import Any
from typing import Dict
from typing import Tuple
from django.db import models
from django.utils.translation import gettext_lazy as _
from . import ShortUUID
def _generate_uuid(self) -> str:
    """Generate a short random string."""
    return self.prefix + ShortUUID(alphabet=self.alphabet).random(length=self.length)