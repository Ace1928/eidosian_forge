from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \

        Perform a scan of DynamoDB.

        :type table: :class:`boto.dynamodb.table.Table`
        :param table: The Table object that is being scanned.

        :type scan_filter: A dict
        :param scan_filter: A dictionary where the key is the
            attribute name and the value is a
            :class:`boto.dynamodb.condition.Condition` object.
            Valid Condition objects include:

             * EQ - equal (1)
             * NE - not equal (1)
             * LE - less than or equal (1)
             * LT - less than (1)
             * GE - greater than or equal (1)
             * GT - greater than (1)
             * NOT_NULL - attribute exists (0, use None)
             * NULL - attribute does not exist (0, use None)
             * CONTAINS - substring or value in list (1)
             * NOT_CONTAINS - absence of substring or value in list (1)
             * BEGINS_WITH - substring prefix (1)
             * IN - exact match in list (N)
             * BETWEEN - >= first value, <= second value (2)

        :type attributes_to_get: list
        :param attributes_to_get: A list of attribute names.
            If supplied, only the specified attribute names will
            be returned.  Otherwise, all attributes will be returned.

        :type request_limit: int
        :param request_limit: The maximum number of items to retrieve
            from Amazon DynamoDB on each request.  You may want to set
            a specific request_limit based on the provisioned throughput
            of your table.  The default behavior is to retrieve as many
            results as possible per request.

        :type max_results: int
        :param max_results: The maximum number of results that will
            be retrieved from Amazon DynamoDB in total.  For example,
            if you only wanted to see the first 100 results from the
            query, regardless of how many were actually available, you
            could set max_results to 100 and the generator returned
            from the query method will only yeild 100 results max.

        :type count: bool
        :param count: If True, Amazon DynamoDB returns a total
            number of items for the Scan operation, even if the
            operation has no matching items for the assigned filter.
            If count is True, the actual items are not returned and
            the count is accessible as the ``count`` attribute of
            the returned object.

        :type exclusive_start_key: list or tuple
        :param exclusive_start_key: Primary key of the item from
            which to continue an earlier query.  This would be
            provided as the LastEvaluatedKey in that query.

        :type item_class: Class
        :param item_class: Allows you to override the class used
            to generate the items. This should be a subclass of
            :class:`boto.dynamodb.item.Item`

        :rtype: :class:`boto.dynamodb.layer2.TableGenerator`
        