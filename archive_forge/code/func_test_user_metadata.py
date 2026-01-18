from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities
def test_user_metadata(tmp_path, infile):
    ref_fn = subdir / 'tweets.20150430-223406.user.csv.ref'
    fields = ['id', 'text', 'user.id', 'user.followers_count', 'user.friends_count']
    outfn = tmp_path / 'tweets.20150430-223406.user.csv'
    json2csv(infile, outfn, fields, gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)