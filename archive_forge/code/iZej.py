import logging
from GameGUI import GameGUI
import asyncio

# Configure logging to the most detailed level possible, ensuring that every possible detail is captured in the log
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename='game_log.log', filemode='a'
)

# Instantiate the GameGUI object, which initializes the game environment and settings with meticulous detail
async def initialize_game():
    try:
        game: GameGUI = GameGUI()  # Type annotation ensures that 'game' is recognized as an instance of GameGUI
        logging.debug(
try:
    game: GameGUI = (
        GameGUI()
    )  # Type annotation ensures that 'game' is recognized as an instance of GameGUI
    logging.debug(
        "GameGUI object has been instantiated successfully, initializing the game environment and settings."
    )
except Exception as instantiation_exception:
    logging.error(
        f"An error occurred during the instantiation of the GameGUI object: {instantiation_exception}",
        exc_info=True,
    )
    raise RuntimeError(
        f"Instantiation of the GameGUI object failed due to: {instantiation_exception}"
    ) from instantiation_exception

# Enter the main loop of the game, which continues as long as the 'running' attribute of the game object is True
while game.running:
    # Display the current menu, which could be the main menu, options, or any other defined menu in the game's GUI system
    logging.debug("Attempting to display the current menu.")
    try:
        game.curr_menu.display_menu()
        logging.debug("Current menu has been displayed successfully.")
    except Exception as display_menu_exception:
        logging.error(
            f"An error occurred while attempting to display the current menu: {display_menu_exception}",
            exc_info=True,
        )
        raise RuntimeError(
            f"Displaying the current menu failed due to: {display_menu_exception}"
        ) from display_menu_exception

    # Execute the main game loop which handles events, updates game state, and renders the game frame by frame
    logging.debug("Entering the game loop.")
    try:
        game.game_loop()
        logging.debug("Game loop execution has completed for the current frame.")
    except Exception as game_loop_exception:
        logging.error(
            f"An error occurred during the game loop execution: {game_loop_exception}",
            exc_info=True,
        )
        raise RuntimeError(
            f"Game loop execution failed due to: {game_loop_exception}"
        ) from game_loop_exception
