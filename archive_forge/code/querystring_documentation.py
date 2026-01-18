
        Create a node

        ### Parameters

        - **children**: One or more sub-conditions. These can be additional
            `intersect`, `disjunct`, `union`, `optional`, or any other `Node`
            type.

            The semantics of multiple conditions are dependent on the type of
            query. For an `intersection` node, this amounts to a logical AND,
            for a `union` node, this amounts to a logical `OR`.

        - **kwparams**: key-value parameters. Each key is the name of a field,
            and the value should be a field value. This can be one of the
            following:

            - Simple string (for text field matches)
            - value returned by one of the helper functions
            - list of either a string or a value


        ### Examples

        Field `num` should be between 1 and 10
        ```
        intersect(num=between(1, 10)
        ```

        Name can either be `bob` or `john`

        ```
        union(name=("bob", "john"))
        ```

        Don't select countries in Israel, Japan, or US

        ```
        disjunct_union(country=("il", "jp", "us"))
        ```
        