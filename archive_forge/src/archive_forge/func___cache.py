import suds.cache
import suds.plugin
import suds.sax.parser
import suds.transport
def __cache(self):
    """
        Get the I{object cache}.

        @return: The I{cache} when I{cachingpolicy} = B{0}.
        @rtype: L{Cache}

        """
    if self.options.cachingpolicy == 0:
        return self.options.cache
    return suds.cache.NoCache()