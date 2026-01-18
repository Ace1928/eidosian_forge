import unittest
import pytest
from nltk import FreqDist
from nltk.lm import NgramCounter
from nltk.util import everygrams
Tests for NgramCounter that only involve lookup, no modification.