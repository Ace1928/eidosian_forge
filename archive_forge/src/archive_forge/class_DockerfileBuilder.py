from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.core import exceptions
class DockerfileBuilder(dataobject.DataObject):
    """Data for a request to build with an existing Dockerfile."""
    NAMES = ('dockerfile',)

    def DockerfileAbsPath(self, context):
        return os.path.abspath(os.path.join(context, self.dockerfile))

    def DockerfileRelPath(self, context):
        return os.path.relpath(self.DockerfileAbsPath(context), context)

    def Validate(self, context):
        complete_path = self.DockerfileAbsPath(context)
        if os.path.commonprefix([context, complete_path]) != context:
            raise InvalidLocationError('Invalid Dockerfile path. Dockerfile must be located in the build context directory.\nDockerfile: {0}\nBuild Context Directory: {1}'.format(complete_path, context))
        if not os.path.exists(complete_path):
            raise InvalidLocationError(complete_path + ' does not exist.')