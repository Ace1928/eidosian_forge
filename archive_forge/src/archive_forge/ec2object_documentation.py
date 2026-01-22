from boto.ec2.tag import TagSet

        Removes tags from this object.  Removing tags involves a round-trip
        to the EC2 service.

        :type tags: dict
        :param tags: A dictionary of key-value pairs for the tags being removed.
                     For each key, the provided value must match the value
                     currently stored in EC2.  If not, that particular tag will
                     not be removed.  However, if a value of None is provided,
                     the tag will be unconditionally deleted.
                     NOTE: There is an important distinction between a value of
                     '' and a value of None.
        