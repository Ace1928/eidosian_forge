import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union
from traitlets.config import Instance, LoggingConfigurable, Unicode
from ..connect import KernelConnectionInfo

        Walks entries in the kernelspec's env stanza and applies substitutions from current env.

        This method is called from `KernelProvisionerBase.pre_launch()` during the kernel's
        start sequence.

        Returns the substituted list of env entries.

        NOTE: This method is private and is not intended to be overridden by provisioners.
        