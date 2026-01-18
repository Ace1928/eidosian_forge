import itertools
from heat.scaling import template
from heat.tests import common
Test case for up-to-date resources in template.

        If some of the old resources already have the new resource definition,
        then they won't be considered for replacement, and the next resource
        that is out-of-date will be replaced.
        