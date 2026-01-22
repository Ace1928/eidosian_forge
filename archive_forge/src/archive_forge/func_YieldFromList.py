from apitools.base.py import encoding
import six
def YieldFromList(service, request, global_params=None, limit=None, batch_size=100, method='List', field='items', predicate=None, current_token_attribute='pageToken', next_token_attribute='nextPageToken', batch_size_attribute='maxResults', get_field_func=_GetattrNested):
    """Make a series of List requests, keeping track of page tokens.

    Args:
      service: apitools_base.BaseApiService, A service with a .List() method.
      request: protorpc.messages.Message, The request message
          corresponding to the service's .List() method, with all the
          attributes populated except the .maxResults and .pageToken
          attributes.
      global_params: protorpc.messages.Message, The global query parameters to
           provide when calling the given method.
      limit: int, The maximum number of records to yield. None if all available
          records should be yielded.
      batch_size: int, The number of items to retrieve per request.
      method: str, The name of the method used to fetch resources.
      field: str, The field in the response that will be a list of items.
      predicate: lambda, A function that returns true for items to be yielded.
      current_token_attribute: str or tuple, The name of the attribute in a
          request message holding the page token for the page being
          requested. If a tuple, path to attribute.
      next_token_attribute: str or tuple, The name of the attribute in a
          response message holding the page token for the next page. If a
          tuple, path to the attribute.
      batch_size_attribute: str or tuple, The name of the attribute in a
          response message holding the maximum number of results to be
          returned. None if caller-specified batch size is unsupported.
          If a tuple, path to the attribute.
      get_field_func: Function that returns the items to be yielded. Argument
          is response message, and field.

    Yields:
      protorpc.message.Message, The resources listed by the service.

    """
    request = encoding.CopyProtoMessage(request)
    _SetattrNested(request, current_token_attribute, None)
    while limit is None or limit:
        if batch_size_attribute:
            if batch_size is None:
                request_batch_size = None
            else:
                request_batch_size = min(batch_size, limit or batch_size)
            _SetattrNested(request, batch_size_attribute, request_batch_size)
        response = getattr(service, method)(request, global_params=global_params)
        items = get_field_func(response, field)
        if predicate:
            items = list(filter(predicate, items))
        for item in items:
            yield item
            if limit is None:
                continue
            limit -= 1
            if not limit:
                return
        token = _GetattrNested(response, next_token_attribute)
        if not token:
            return
        _SetattrNested(request, current_token_attribute, token)