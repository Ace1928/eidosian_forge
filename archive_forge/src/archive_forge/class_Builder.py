class Builder:

    def __init__(self):
        self.ourIdentityKey = None
        self.ourBaseKey = None
        self.theirIdentityKey = None
        self.theirSignedPreKey = None
        self.theirRatchetKey = None
        self.theirOneTimePreKey = None

    def setOurIdentityKey(self, ourIdentityKey):
        self.ourIdentityKey = ourIdentityKey
        return self

    def setOurBaseKey(self, ourBaseKey):
        self.ourBaseKey = ourBaseKey
        return self

    def setTheirRatchetKey(self, theirRatchetKey):
        self.theirRatchetKey = theirRatchetKey
        return self

    def setTheirIdentityKey(self, theirIdentityKey):
        self.theirIdentityKey = theirIdentityKey
        return self

    def setTheirSignedPreKey(self, theirSignedPreKey):
        self.theirSignedPreKey = theirSignedPreKey
        return self

    def setTheirOneTimePreKey(self, theirOneTimePreKey):
        self.theirOneTimePreKey = theirOneTimePreKey
        return self

    def create(self):
        return AliceAxolotlParameters(self.ourIdentityKey, self.ourBaseKey, self.theirIdentityKey, self.theirSignedPreKey, self.theirRatchetKey, self.theirOneTimePreKey)