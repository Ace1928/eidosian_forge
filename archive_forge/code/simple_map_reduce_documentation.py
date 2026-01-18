import logging
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow
from taskflow.patterns import unordered_flow
from taskflow import task
Yields back chunk size pieces from zero to upperbound - 1.