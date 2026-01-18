from apitools.base.py import encoding
import six
Make a series of List requests, keeping track of page tokens.

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

    