import random
import email.message
import pyzor
def add_digest(self, digest):
    self.add_header('Op-Digest', digest)
    self.digest_count += 1