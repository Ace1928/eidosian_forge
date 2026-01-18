from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities
def test_tweet_usermention(tmp_path, infile):
    ref_fn = subdir / 'tweets.20150430-223406.usermention.csv.ref'
    outfn = tmp_path / 'tweets.20150430-223406.usermention.csv'
    json2csv_entities(infile, outfn, ['id', 'text'], 'user_mentions', ['id', 'screen_name'], gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)