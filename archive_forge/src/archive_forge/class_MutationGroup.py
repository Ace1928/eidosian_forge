from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MutationGroup(_messages.Message):
    """A group of mutations to be committed together. Related mutations should
  be placed in a group. For example, two mutations inserting rows with the
  same primary key prefix in both parent and child tables are related.

  Fields:
    mutations: Required. The mutations in this group.
  """
    mutations = _messages.MessageField('Mutation', 1, repeated=True)