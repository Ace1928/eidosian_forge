from tests.compat import unittest
import boto.cloudfront as cf

        Test that wildcards are retained as literals
        See: http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Invalidation.html#invalidation-specifying-objects-paths
        