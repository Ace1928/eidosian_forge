from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities

    Sanity check that file comparison is not giving false positives.
    