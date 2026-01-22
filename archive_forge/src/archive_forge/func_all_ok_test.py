import subprocess, os, sys, re, difflib
def all_ok_test(suite, *args):
    for results in args:
        assert 'Ran 36 tests' in results
        assert 'OK' in results