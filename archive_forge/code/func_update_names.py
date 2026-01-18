from collections import OrderedDict
def update_names(tensor, names, rename_map, inplace):
    """There are two usages:

    tensor.rename(*names) returns a view on tensor with named dims `names`.
    `names` must be of length `tensor.dim()`; otherwise, if '...' is in `names`,
    then it is expanded greedily to be equal to the corresponding names from
    `tensor.names`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename('...', 'height', 'width').names
    ('N', 'C', 'height', 'width')

    >>> # xdoctest: +SKIP
    >>> x.rename('batch', '...', 'width').names
    ('batch', 'C', 'H', 'width')

    ```

    tensor.rename(**rename_map) returns a view on tensor that has rename dims
        as specified in the mapping `rename_map`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename(W='width', H='height').names
    ('N', 'C', 'height', 'width')

    ```

    Finally, tensor.rename has an in-place version called tensor.rename_.
    """
    has_names = len(names) > 0
    has_rename_pairs = bool(rename_map)
    if has_names and has_rename_pairs:
        raise RuntimeError(f'{namer_api_name(inplace)}: This function takes either positional args or keyword args, but not both. Use tensor.{namer_api_name(inplace)}(*names) to name dims and tensor.{namer_api_name(inplace)}(**rename_map) to rename dims.')
    if not has_names and (not has_rename_pairs):
        return update_names_with_list(tensor, names, inplace)
    if has_names:
        return update_names_with_list(tensor, names, inplace)
    return update_names_with_mapping(tensor, rename_map, inplace)