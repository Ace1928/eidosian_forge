from typing import List
from redis import DataError

        Create Vector Field. Notice that Vector cannot have sortable or no_index tag,
        although it's also a Field.

        ``name`` is the name of the field.

        ``algorithm`` can be "FLAT" or "HNSW".

        ``attributes`` each algorithm can have specific attributes. Some of them
        are mandatory and some of them are optional. See
        https://oss.redis.com/redisearch/master/Vectors/#specific_creation_attributes_per_algorithm
        for more information.
        