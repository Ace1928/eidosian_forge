import sys
import os.path
from os import path
from polib import *

fileName = sys.argv[1]

if path.exists(fileName):

    pofiledata = pofile(fileName)

    entries: list[POEntry] = pofiledata
    words = []
    for entry in entries:
        msgid: str = entry.msgid
        msgid = msgid.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        for word in msgid.split(' '):
            if len(word) > 0 and not str.isspace(word):
                words.append(word)

    print("Input file:    " + os.path.basename(fileName))
    print("String count:  " + str(len(entries)))
    print("Word count:    " + str(len(words)))

    with open('words.txt', 'w') as f:

        f.write("[Stats]\n")
        f.write("Input file:    " + os.path.basename(fileName) + "\n")
        f.write("String count:  " + str(len(entries)) + "\n")
        f.write("Word count:    " + str(len(words)) + "\n")
        f.write("\n")
        f.write("[Words Found]\n")
        for word in words:
            f.write(word+"\n")
        