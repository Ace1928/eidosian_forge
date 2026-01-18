import time
import hashlib
import sys
import gc
import io
import collections
import itertools
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from joblib.hashing import hash
from joblib.func_inspect import filter_args
from joblib.memory import Memory
from joblib.testing import raises, skipif, fixture, parametrize
from joblib.test.common import np, with_numpy
def test_set_hash(tmpdir):
    k = KlassWithCachedMethod(tmpdir.strpath)
    s = set(['#s12069__c_maps.nii.gz', '#s12158__c_maps.nii.gz', '#s12258__c_maps.nii.gz', '#s12277__c_maps.nii.gz', '#s12300__c_maps.nii.gz', '#s12401__c_maps.nii.gz', '#s12430__c_maps.nii.gz', '#s13817__c_maps.nii.gz', '#s13903__c_maps.nii.gz', '#s13916__c_maps.nii.gz', '#s13981__c_maps.nii.gz', '#s13982__c_maps.nii.gz', '#s13983__c_maps.nii.gz'])
    a = k.f(s)
    b = k.f(a)
    assert hash(a) == hash(b)