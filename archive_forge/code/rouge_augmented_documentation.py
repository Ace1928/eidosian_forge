import nltk
import os
import re
import itertools
import collections
import pkg_resources
from rouge import Rouge

        Compute precision, recall and f1 score between hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between hypothesis and references

        Raises:
          ValueError: raises exception if a type of hypothesis is different than the one of reference
          ValueError: raises exception if a len of hypothesis is different than the one of reference
        