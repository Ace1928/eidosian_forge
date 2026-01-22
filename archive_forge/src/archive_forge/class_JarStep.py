from boto.compat import six
class JarStep(Step):
    """
    Custom jar step
    """

    def __init__(self, name, jar, main_class=None, action_on_failure='TERMINATE_JOB_FLOW', step_args=None):
        """
        A elastic mapreduce step that executes a jar

        :type name: str
        :param name: The name of the step
        :type jar: str
        :param jar: S3 URI to the Jar file
        :type main_class: str
        :param main_class: The class to execute in the jar
        :type action_on_failure: str
        :param action_on_failure: An action, defined in the EMR docs to
            take on failure.
        :type step_args: list(str)
        :param step_args: A list of arguments to pass to the step
        """
        self.name = name
        self._jar = jar
        self._main_class = main_class
        self.action_on_failure = action_on_failure
        if isinstance(step_args, six.string_types):
            step_args = [step_args]
        self.step_args = step_args

    def jar(self):
        return self._jar

    def args(self):
        args = []
        if self.step_args:
            args.extend(self.step_args)
        return args

    def main_class(self):
        return self._main_class