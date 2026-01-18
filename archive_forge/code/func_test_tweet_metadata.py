from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities
def test_tweet_metadata(tmp_path, infile):
    ref_fn = subdir / 'tweets.20150430-223406.tweet.csv.ref'
    fields = ['created_at', 'favorite_count', 'id', 'in_reply_to_status_id', 'in_reply_to_user_id', 'retweet_count', 'retweeted', 'text', 'truncated', 'user.id']
    outfn = tmp_path / 'tweets.20150430-223406.tweet.csv'
    json2csv(infile, outfn, fields, gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)