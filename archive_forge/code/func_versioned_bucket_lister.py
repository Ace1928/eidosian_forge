def versioned_bucket_lister(bucket, prefix='', delimiter='', marker='', generation_marker='', headers=None):
    """
    A generator function for listing versioned objects.
    """
    more_results = True
    k = None
    while more_results:
        rs = bucket.get_all_versions(prefix=prefix, marker=marker, generation_marker=generation_marker, delimiter=delimiter, headers=headers, max_keys=999)
        for k in rs:
            yield k
        marker = rs.next_marker
        generation_marker = rs.next_generation_marker
        more_results = rs.is_truncated