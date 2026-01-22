from collections import defaultdict
from logging import getLogger
from typing import Any, DefaultDict
from pip._vendor.resolvelib.reporters import BaseReporter
from .base import Candidate, Requirement
def adding_requirement(self, requirement: Requirement, parent: Candidate) -> None:
    logger.info('Reporter.adding_requirement(%r, %r)', requirement, parent)