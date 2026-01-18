from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_operations_checker(self):
    tests = [('all allowed', checkers.allow_caveat(['op1', 'op2', 'op4', 'op3']), ['op1', 'op3', 'op2'], None), ('none denied', checkers.deny_caveat(['op1', 'op2']), ['op3', 'op4'], None), ('one not allowed', checkers.allow_caveat(['op1', 'op2']), ['op1', 'op3'], 'caveat "allow op1 op2" not satisfied: op3 not allowed'), ('one not denied', checkers.deny_caveat(['op1', 'op2']), ['op4', 'op5', 'op2'], 'caveat "deny op1 op2" not satisfied: op2 not allowed'), ('no operations, allow caveat', checkers.allow_caveat(['op1']), [], 'caveat "allow op1" not satisfied: op1 not allowed'), ('no operations, deny caveat', checkers.deny_caveat(['op1']), [], None), ('no operations, empty allow caveat', checkers.Caveat(condition=checkers.COND_ALLOW), [], 'caveat "allow" not satisfied: no operations allowed')]
    checker = checkers.Checker()
    for test in tests:
        print(test[0])
        ctx = checkers.context_with_operations(checkers.AuthContext(), test[2])
        err = checker.check_first_party_caveat(ctx, test[1].condition)
        if test[3] is None:
            self.assertIsNone(err)
            continue
        self.assertEqual(err, test[3])