
    Provide information about attributes for an index field.
    A maximum of 20 source attributes can be configured for
    each index field.

    :ivar default: Optional default value if the source attribute
        is not specified in a document.
        
    :ivar name: The name of the document source field to add
        to this ``IndexField``.

    :ivar data_function: Identifies the transformation to apply
        when copying data from a source attribute.
        
    :ivar data_map: The value is a dict with the following keys:
        * cases - A dict that translates source field values
            to custom values.
        * default - An optional default value to use if the
            source attribute is not specified in a document.
        * name - the name of the document source field to add
            to this ``IndexField``
    :ivar data_trim_title: Trims common title words from a source
        document attribute when populating an ``IndexField``.
        This can be used to create an ``IndexField`` you can
        use for sorting.  The value is a dict with the following
        fields:
        * default - An optional default value.
        * language - an IETF RFC 4646 language code.
        * separator - The separator that follows the text to trim.
        * name - The name of the document source field to add.
    