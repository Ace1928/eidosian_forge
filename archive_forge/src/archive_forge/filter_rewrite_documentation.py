from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_expr_rewrite
Always quote the operand as the Cloud Filter Library won't be able to parse as values all arbitrary strings.