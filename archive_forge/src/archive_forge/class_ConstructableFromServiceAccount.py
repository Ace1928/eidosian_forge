from google.oauth2 import service_account
class ConstructableFromServiceAccount:

    @classmethod
    def from_service_account_file(cls, filename, **kwargs):
        f'Creates an instance of this client using the provided credentials file.\n        Args:\n            filename (str): The path to the service account private key json\n                file.\n            kwargs: Additional arguments to pass to the constructor.\n        Returns:\n            A {cls.__name__}.\n        '
        credentials = service_account.Credentials.from_service_account_file(filename)
        return cls(credentials=credentials, **kwargs)
    from_service_account_json = from_service_account_file