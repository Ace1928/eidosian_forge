from boto.compat import six
class ScriptRunnerStep(JarStep):
    ScriptRunnerJar = 's3n://us-east-1.elasticmapreduce/libs/script-runner/script-runner.jar'

    def __init__(self, name, **kw):
        super(ScriptRunnerStep, self).__init__(name, self.ScriptRunnerJar, **kw)