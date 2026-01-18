import csv
import logging
import os
import random
import sys
import futurist
from taskflow import engines
from taskflow.patterns import unordered_flow as uf
from taskflow import task
Performs a modification of an input row, creating a output row.