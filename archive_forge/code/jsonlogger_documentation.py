import logging
import json
import re
from datetime import date, datetime, time, timezone
import traceback
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from inspect import istraceback
from collections import OrderedDict
Formats a log record and serializes to json