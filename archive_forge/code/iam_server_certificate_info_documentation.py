from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
Retrieve the attributes of a server certificate if it exists or all certs.
    Args:
        iam (botocore.client.IAM): The boto3 iam instance.

    Kwargs:
        name (str): The name of the server certificate.

    Basic Usage:
        >>> import boto3
        >>> iam = boto3.client('iam')
        >>> name = "server-cert-name"
        >>> results = get_server_certs(iam, name)
        {
            "upload_date": "2015-04-25T00:36:40+00:00",
            "server_certificate_id": "ADWAJXWTZAXIPIMQHMJPO",
            "certificate_body": "-----BEGIN CERTIFICATE-----
bunch of random data
-----END CERTIFICATE-----",
            "server_certificate_name": "server-cert-name",
            "expiration": "2017-06-15T12:00:00+00:00",
            "path": "/",
            "arn": "arn:aws:iam::123456789012:server-certificate/server-cert-name"
        }
    