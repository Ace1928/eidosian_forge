from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (

This is default deletion operation.
Delete configuration if there is no non-key leaf, and
delete non-key leaf configuration, if any, if the values of non-key leaf are
equal between command and existing configuration.
