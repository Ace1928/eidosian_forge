from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import traffic
@property
def vpc_connector(self):
    return self.annotations.get(u'run.googleapis.com/vpc-access-connector')