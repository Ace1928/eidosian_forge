import inspect
import sys
from typing import Dict, List, Set, Tuple
from wandb.errors import UsageError
from wandb.sdk.wandb_settings import Settings
import sys
from typing import Tuple
Return the order in which settings should be modified, based on dependencies.