from boto.compat import six
class HiveStep(HiveBase):
    """
    Hive script step
    """

    def __init__(self, name, hive_file, hive_versions='latest', hive_args=None):
        step_args = []
        step_args.extend(self.BaseArgs)
        step_args.extend(['--hive-versions', hive_versions])
        step_args.extend(['--run-hive-script', '--args', '-f', hive_file])
        if hive_args is not None:
            step_args.extend(hive_args)
        super(HiveStep, self).__init__(name, step_args=step_args)