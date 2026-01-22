from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test.ios import catalog_manager
from googlecloudsdk.calliope import exceptions
def Prepare(self, args):
    """Load, apply defaults, and perform validation on test arguments.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        gcloud command invocation (i.e. group and command arguments combined).
        Arg values from an optional arg-file and/or arg default values may be
        added to this argparse namespace.

    Raises:
      InvalidArgumentException: If an argument name is unknown, an argument does
        not contain a valid value, or an argument is not valid when used with
        the given type of test.
      RequiredArgumentException: If a required arg is missing.
    """
    all_test_args_set = arg_util.GetSetOfAllTestArgs(self._typed_arg_rules, self._shared_arg_rules)
    args_from_file = arg_file.GetArgsFromArgFile(args.argspec, all_test_args_set)
    arg_util.ApplyLowerPriorityArgs(args, args_from_file, True)
    test_type = self.GetTestTypeOrRaise(args)
    typed_arg_defaults = self._typed_arg_rules[test_type]['defaults']
    shared_arg_defaults = self._shared_arg_rules['defaults']
    arg_util.ApplyLowerPriorityArgs(args, typed_arg_defaults)
    arg_util.ApplyLowerPriorityArgs(args, shared_arg_defaults)
    arg_validate.ValidateArgsForTestType(args, test_type, self._typed_arg_rules, self._shared_arg_rules, all_test_args_set)
    arg_validate.ValidateDeviceList(args, self._catalog_mgr)
    arg_validate.ValidateXcodeVersion(args, self._catalog_mgr)
    arg_validate.ValidateResultsBucket(args)
    arg_validate.ValidateResultsDir(args)
    arg_validate.ValidateScenarioNumbers(args)
    arg_validate.ValidateIosDirectoriesToPullList(args)