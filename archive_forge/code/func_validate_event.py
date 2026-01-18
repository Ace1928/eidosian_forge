import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def validate_event(self, original_list, operation):
    """
        Validate the event arising from a particular TraitList operation.

        Given a test list and an operation to perform, perform
        that operation on both a plain Python list and the corresponding
        TraitList, then:

        - check that the resulting lists match
        - check that the event information generated (if any) is suitably
          normalized
        - check that the list operation can be reconstructed from the
          event information

        Parameters
        ----------
        original_list : list
            List to use for testing.
        operation : callable
            Single-argument callable which accepts the list and performs
            the desired operation on it.

        Raises
        ------
        self.failureException
            If any aspect of the behaviour is found to be incorrect.
        """
    notifications = []

    def notifier(trait_list, index, removed, added):
        notifications.append((index, removed, added))
    python_list = original_list.copy()
    try:
        python_result = operation(python_list)
    except Exception as e:
        python_exception = e
        python_raised = True
    else:
        python_raised = False
    trait_list = TraitList(original_list, notifiers=[notifier])
    try:
        trait_result = operation(trait_list)
    except Exception as e:
        trait_exception = e
        trait_raised = True
    else:
        trait_raised = False
    self.assertEqual(python_list, trait_list)
    self.assertEqual(python_raised, trait_raised)
    if python_raised:
        self.assertEqual(type(python_exception), type(trait_exception))
        return
    self.assertEqual(python_result, trait_result)
    if notifications == []:
        self.assertEqual(trait_list, original_list)
        return
    self.assertEqual(len(notifications), 1)
    index, removed, added = notifications[0]
    self.assertTrue(len(removed) > 0 or len(added) > 0, 'a notification was generated, but no elements were added or removed')
    self.check_index_normalized(index, len(original_list))
    reconstructed = original_list.copy()
    if isinstance(index, slice):
        self.assertEqual(removed, reconstructed[index])
        if added:
            reconstructed[index] = added
        else:
            del reconstructed[index]
    else:
        removed_slice = slice(index, index + len(removed))
        self.assertEqual(removed, reconstructed[removed_slice])
        reconstructed[removed_slice] = added
    self.assertEqual(reconstructed, trait_list)