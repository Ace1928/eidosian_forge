class DeferredProviderLookup(object):

    def __init__(self, api, method):
        self.__api = api
        self.__method = method

    def __get__(self, instance, owner):
        api = getattr(ProviderAPIs, self.__api)
        return getattr(api, self.__method)